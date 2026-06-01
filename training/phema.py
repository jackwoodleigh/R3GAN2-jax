"""Routines for post-hoc EMA and power function EMA proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import copy
import numpy as np
import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

#----------------------------------------------------------------------------
# Convert power function exponent to relative standard deviation
# according to Equation 123.

def exp_to_std(exp):
    exp = np.float64(exp)
    std = np.sqrt((exp + 1) / (exp + 2) ** 2 / (exp + 3))
    return std


#----------------------------------------------------------------------------
# Convert relative standard deviation to power function exponent
# according to Equation 126 and Algorithm 2.

def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp


#----------------------------------------------------------------------------
# Construct response functions for the given EMA profiles
# according to Equations 121 and 108.

def power_function_response(ofs, std, len, axis=0):
    ofs, std = np.broadcast_arrays(ofs, std)
    ofs = np.stack([np.float64(ofs)], axis=axis)
    exp = np.stack([std_to_exp(std)], axis=axis)
    s = [1] * exp.ndim
    s[axis] = -1
    t = np.arange(len).reshape(s)
    resp = np.where(t <= ofs, (t / ofs) ** exp, 0) / ofs * (exp + 1)
    resp = resp / np.sum(resp, axis=axis, keepdims=True)
    return resp

#----------------------------------------------------------------------------
# Compute inner products between the given pairs of EMA profiles
# according to Equation 151 and Algorithm 3.

def power_function_correlation(a_ofs, a_std, b_ofs, b_std):
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    num = (a_exp + 1) * (b_exp + 1) * t_ratio ** t_exp
    den = (a_exp + b_exp + 1) * t_max
    return num / den

#----------------------------------------------------------------------------
# Calculate beta for tracking a given EMA profile during training
# according to Equation 127.

def power_function_beta(std, t_next, t_delta):
    beta = (1 - t_delta / t_next) ** (std_to_exp(std) + 1)
    return beta

#----------------------------------------------------------------------------
# Solve the coefficients for post-hoc EMA reconstruction
# according to Algorithm 3.

def solve_posthoc_coefficients(in_ofs, in_std, out_ofs, out_std): # => [in, out]
    in_ofs, in_std = np.broadcast_arrays(in_ofs, in_std)
    out_ofs, out_std = np.broadcast_arrays(out_ofs, out_std)
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = power_function_correlation(rv(in_ofs), rv(in_std), cv(in_ofs), cv(in_std))
    B = power_function_correlation(rv(in_ofs), rv(in_std), cv(out_ofs), cv(out_std))
    X = np.linalg.solve(A, B)
    X = X / np.sum(X, axis=0)
    return X


#----------------------------------------------------------------------------
# Sharding helpers.

def _make_pick_sharding(mesh, axis_name, n):
    """Return a per-leaf sharding chooser: shard the largest axis whose length
    is divisible by `n`; otherwise replicate. Elementwise ops are invariant to
    which axis we pick, so this is purely a memory-balance heuristic — each
    local chip ends up holding ~1/n of every shardable parameter. Small 1D
    tensors (biases, norms) fall back to replication, which is cheap."""
    def pick(leaf):
        if not hasattr(leaf, 'ndim') or leaf.ndim == 0:
            return NamedSharding(mesh, P())
        best_ax, best_sz = -1, -1
        for ax in range(leaf.ndim):
            if leaf.shape[ax] % n == 0 and leaf.shape[ax] > best_sz:
                best_ax, best_sz = ax, leaf.shape[ax]
        if best_ax < 0:
            return NamedSharding(mesh, P())
        spec = [None] * leaf.ndim
        spec[best_ax] = axis_name
        return NamedSharding(mesh, P(*spec))
    return pick


def _ema_update_impl(ema_states, net_state, betas):
    """Pure function; jitted per-instance in PowerFunctionEMA.__init__ with
    explicit in/out shardings. The math is identical to the unsharded version:
    each leaf gets e * beta_i + n * (1 - beta_i). Elementwise ops compose
    under sharding, so the output is bit-equivalent modulo reduction order
    (there are no reductions here — this is purely pointwise)."""
    return [
        jax.tree_util.tree_map(
            lambda e, n: e * betas[i] + n * (1 - betas[i]),
            es, net_state,
        )
        for i, es in enumerate(ema_states)
    ]

import time
#----------------------------------------------------------------------------
# Class for tracking power function EMA during the training.

class PowerFunctionEMA:
    def __init__(self, net, stds=[0.010, 0.050, 0.100]):
        self.net = net
        self.stds = stds
        self.graphdef, state = nnx.split(net)

        # Per-host mesh. Matches how state_G is device_put_replicated across
        # local_devices() in training_loop.py — each host keeps its own
        # identical copy of the EMA, just like before. No cross-host traffic.
        devs = jax.local_devices()
        self.n_shards = len(devs)
        self.mesh = Mesh(np.array(devs), ('s',))
        self._rep = NamedSharding(self.mesh, P())

        # Per-leaf shardings: shard across 's' when possible, else replicate.
        pick = _make_pick_sharding(self.mesh, 's', self.n_shards)
        self.ema_sharding = jax.tree_util.tree_map(pick, state)
        # Matching replicated tree for the net_state arg.
        self._rep_tree = jax.tree_util.tree_map(lambda _: self._rep, state)

        # Place initial EMA copies. jax.device_put with a sharding both copies
        # and scatters; replaces the old tree_map(lambda x: x.copy(), state).
        self.ema_states = [
            jax.tree_util.tree_map(
                lambda x, s: jax.device_put(x, s),
                state, self.ema_sharding,
            )
            for _ in stds
        ]

        # Bind the update fn with explicit shardings so XLA doesn't
        # silently reshape or gather. Compiles on first call.
        ema_list_sharding = [self.ema_sharding for _ in stds]
        self._update_fn = jax.jit(
            _ema_update_impl,
            in_shardings=(ema_list_sharding, self._rep_tree, self._rep),
            out_shardings=ema_list_sharding,
        )

    def reset(self):
        _, net_state = nnx.split(self.net)
        self.ema_states = [
            jax.tree_util.tree_map(
                lambda x, s: jax.device_put(x, s),
                net_state, self.ema_sharding,
            )
            for _ in self.stds
        ]   
    

    def update(self, net_state, cur_nimg, batch_size):
        # net_state comes in on device 0 (from state_G[0] in training_loop).
        # jit will reshard it to replicated across the local mesh via ICI,
        # which is intra-host and fast (~hundreds of GB/s).
        betas = jnp.array([
            float(power_function_beta(std=std, t_next=cur_nimg, t_delta=batch_size))
            for std in self.stds
        ])
        self.ema_states = self._update_fn(self.ema_states, net_state, betas)

    def get(self):
        # nnx.merge returns a module whose params carry the EMA's sharding;
        # BatchStat comes from the live net and is unchanged from before.
        bn_state = nnx.state(self.net, nnx.BatchStat)
        results = []
        for std, ema_state in zip(self.stds, self.ema_states):
            merged = nnx.merge(self.graphdef, ema_state)
            nnx.update(merged, bn_state)
            results.append((merged, f'-{std:.5f}'))
        return results

    def state_dict(self):
        return {'stds': self.stds, 'emas': self.ema_states}

    def load_state_dict(self, state):
        self.stds = state['stds']
        # Re-place loaded states onto our sharding. Handles both numpy (from a
        # resumed checkpoint on host) and differently-placed jax arrays.
        self.ema_states = [
            jax.tree_util.tree_map(
                lambda x, s: jax.device_put(x, s),
                ema, self.ema_sharding,
            )
            for ema in state['emas']
        ]

    @property
    def emas(self):
        return self.ema_states


#----------------------------------------------------------------------------
# Class for tracking traditional EMA during training.

class TraditionalEMA:
    def __init__(self, net, halflife_Mimg=float('inf'), rampup_ratio=0.09):
        self.net = net
        self.halflife_Mimg = halflife_Mimg
        self.rampup_ratio = rampup_ratio
        graphdef, state = nnx.split(net)
        self.ema = nnx.merge(graphdef, jax.tree_util.tree_map(lambda x: x.copy(), state))

    def reset(self):
        net_state = nnx.state(self.net, nnx.Param)
        nnx.update(self.ema, net_state)

    def update(self, cur_nimg, batch_size):
        halflife_Mimg = self.halflife_Mimg
        if self.rampup_ratio is not None:
            halflife_Mimg = min(halflife_Mimg, cur_nimg / 1e6 * self.rampup_ratio)
        beta = 0.5 ** (batch_size / max(halflife_Mimg * 1e6, 1e-8))
        net_state = nnx.state(self.net, nnx.Param)
        ema_state = nnx.state(self.ema, nnx.Param)
        new_state = jax.tree_util.tree_map(
            lambda e, n: e * beta + n * (1 - beta),
            ema_state, net_state
        )
        nnx.update(self.ema, new_state)

    def get(self):
        net_state = nnx.state(self.net, nnx.BatchStat)
        nnx.update(self.ema, net_state)
        return self.ema
