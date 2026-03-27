"""Routines for post-hoc EMA and power function EMA proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import copy
import numpy as np
import jax
from flax import nnx
from jax import numpy as jnp
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
# Class for tracking power function EMA during the training.

@jax.jit
def _ema_update_all(ema_states, net_state, betas):
    """Single fused kernel for all EMA updates."""
    return [
        jax.tree_util.tree_map(
            lambda e, n: e * betas[i] + n * (1 - betas[i]),
            es, net_state,
        )
        for i, es in enumerate(ema_states)
    ]
 
class PowerFunctionEMA:
    def __init__(self, net, stds=[0.010, 0.050, 0.100]):
        self.net = net
        self.stds = stds
        self.graphdef, state = nnx.split(net)
        self.ema_states = [jax.tree_util.tree_map(lambda x: x.copy(), state) for _ in stds]
 
    def reset(self):
        _, net_state = nnx.split(self.net)
        self.ema_states = [jax.tree_util.tree_map(lambda x: x.copy(), net_state) for _ in self.stds]
 
    def update(self, net_state, cur_nimg, batch_size):
        betas = jnp.array([
            float(power_function_beta(std=std, t_next=cur_nimg, t_delta=batch_size))
            for std in self.stds
        ])
        self.ema_states = _ema_update_all(self.ema_states, net_state, betas)
 
    def get(self):
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
        self.ema_states = state['emas']
    
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