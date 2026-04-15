import jax
import jax.numpy as jnp
from jax import lax, nn as jnn
from flax import nnx
import optax
from functools import partial
from .MagnitudePreservingLayers import NormalizedParam, CenteredNormalizedParam, Normalize

def normalize_state_weights(state):
    def fn(x):
        if not isinstance(x, nnx.VariableState):
            return x
        if x.type is NormalizedParam:
            w = Normalize(x.value.astype(jnp.float32))
            return x.replace(value=w.astype(x.value.dtype))
        if x.type is CenteredNormalizedParam:
            w = x.value.astype(jnp.float32)
            w = w - jnp.mean(w, axis=list(range(1, w.ndim)), keepdims=True)
            w = Normalize(w)
            return x.replace(value=w.astype(x.value.dtype))
        return x
    return jax.tree.map(fn, state, is_leaf=lambda x: isinstance(x, nnx.VariableState))

def ZeroCenteredGradientPenalty(vjp_fn, FakeLogits, RealLogits):
    R1grads = vjp_fn((jnp.zeros_like(FakeLogits), jnp.ones_like(RealLogits)))[1]
    R2grads = vjp_fn((jnp.ones_like(FakeLogits), jnp.zeros_like(RealLogits)))[0] 
    R1Penalty = (R1grads ** 2).sum(axis=(1, 2, 3))
    R2Penalty = (R2grads ** 2).sum(axis=(1, 2, 3))
    return R1Penalty, R2Penalty

def FusedZeroCenteredGradientPenalty(vjp_fn, FakeLogits, RealLogits):
    R2grads, R1grads = vjp_fn((jnp.ones_like(FakeLogits), jnp.ones_like(RealLogits)))
    R1Penalty = (R1grads ** 2).sum(axis=(1, 2, 3))
    R2Penalty = (R2grads ** 2).sum(axis=(1, 2, 3))
    return R1Penalty, R2Penalty

def loss_G(graphdef_G, graphdef_D, augment, state_G, state_D, real_img, z, c, cur_gamma, cur_aug_p, key):
    key = jax.random.fold_in(key, ord('G'))
    G = nnx.merge(graphdef_G, state_G)
    D = nnx.merge(graphdef_D, state_D)
    
    @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
    def forward(z, real_img):
        FakeSamples = G(z, c, key)
        TransformedFake, TransformedReal = augment([FakeSamples, real_img], cur_aug_p, key)
        FakeLogits = D(TransformedFake, c)
        RealLogits = D(TransformedReal, c)
        RelativisticLogits = FakeLogits - RealLogits
        return jnn.softplus(-RelativisticLogits).mean()

    return forward(z, real_img)

def loss_D(graphdef_G, graphdef_D, augment, state_D, state_G, RealSamples, z, c, cur_gamma, cur_aug_p, key):
    key = jax.random.fold_in(key, ord('D'))
    G = nnx.merge(graphdef_G, state_G)
    D = nnx.merge(graphdef_D, state_D)
    FakeSamples = lax.stop_gradient(G(z, c, key))
    TransformedFake, TransformedReal = augment([FakeSamples, RealSamples], cur_aug_p, key)
    FakeLogits = D(TransformedFake, c)
    RealLogits = D(TransformedReal, c)
    RelativisticLogits = RealLogits - FakeLogits
    return jnn.softplus(-RelativisticLogits).mean()

def loss_D_gp(graphdef_G, graphdef_D, augment, state_D, state_G, RealSamples, z, c, cur_gamma, cur_aug_p, key):
    key = jax.random.fold_in(key, ord('D'))
    G = nnx.merge(graphdef_G, state_G)
    D = nnx.merge(graphdef_D, state_D)
    FakeSamples = lax.stop_gradient(G(z, c, key))

    def joint_forward(fake, real):
        TransformedFake, TransformedReal = augment([fake, real], cur_aug_p, key)
        return D(TransformedFake, c), D(TransformedReal, c)

    (FakeLogits, RealLogits), vjp_fn = jax.vjp(jax.checkpoint(joint_forward, policy=jax.checkpoint_policies.nothing_saveable), FakeSamples, RealSamples)
    R1Penalty, R2Penalty = FusedZeroCenteredGradientPenalty(vjp_fn, FakeLogits, RealLogits)
    RelativisticLogits = RealLogits - FakeLogits
    AdversarialLoss = jnn.softplus(-RelativisticLogits)
    return (AdversarialLoss + (cur_gamma / 2) * (R1Penalty + R2Penalty)).mean()

def make_step(loss_fn, tx):
    @partial(jax.pmap, axis_name='batch')
    def step(state, state_other, opt_state, real_imgs, noises, conditions, cur_gamma, cur_aug_p, key):
        indices = jnp.arange(real_imgs.shape[0])

        def scan_fn(acc_grads, chunk):
            real_img, z, c, idx = chunk
            chunk_key = jax.random.fold_in(key, idx)
            loss, grads = jax.value_and_grad(loss_fn)(state, state_other, real_img, z, c, cur_gamma, cur_aug_p, chunk_key)
            return jax.tree.map(lambda a, b: a + b, acc_grads, grads), loss

        init_grads = jax.tree.map(jnp.zeros_like, state)
        acc_grads, losses = jax.lax.scan(scan_fn, init_grads, (real_imgs, noises, conditions, indices))
        acc_grads = jax.tree.map(lambda g: g / real_imgs.shape[0], acc_grads)
        acc_grads = lax.pmean(acc_grads, axis_name='batch')
        updates, new_opt_state = tx.update(acc_grads, opt_state)
        
        # Weight normalization 
        return normalize_state_weights(optax.apply_updates(state, updates)), new_opt_state, jnp.mean(losses)
    return step

def make_step_gp(loss_fn, tx, gp_micro_batch=8):
    @partial(jax.pmap, axis_name='batch')
    def step(state, state_other, opt_state, real_imgs, noises, conditions, cur_gamma, cur_aug_p, key):
        # Rechunk: [n_chunks, big_chunk, ...] -> [more_chunks, gp_micro_batch, ...]
        def rechunk(x):
            total = x.shape[0] * x.shape[1]
            return x.reshape((total // gp_micro_batch, gp_micro_batch) + x.shape[2:])

        real_imgs = rechunk(real_imgs)
        noises = rechunk(noises)
        conditions = rechunk(conditions)
        indices = jnp.arange(real_imgs.shape[0])

        def scan_fn(acc_grads, chunk):
            real_img, z, c, idx = chunk
            chunk_key = jax.random.fold_in(key, idx)
            loss, grads = jax.value_and_grad(loss_fn)(state, state_other, real_img, z, c, cur_gamma, cur_aug_p, chunk_key)
            return jax.tree.map(lambda a, b: a + b, acc_grads, grads), loss

        init_grads = jax.tree.map(jnp.zeros_like, state)
        acc_grads, losses = jax.lax.scan(scan_fn, init_grads, (real_imgs, noises, conditions, indices))
        acc_grads = jax.tree.map(lambda g: g / real_imgs.shape[0], acc_grads)
        acc_grads = lax.pmean(acc_grads, axis_name='batch')
        updates, new_opt_state = tx.update(acc_grads, opt_state)
        return normalize_state_weights(optax.apply_updates(state, updates)), new_opt_state, jnp.mean(losses)
    return step

def build_train_steps(graphdef_G, graphdef_D, tx_G, tx_D, augment_pipe=None, reg_interval=16, gp_micro_batch=8):
    augment = augment_pipe if augment_pipe is not None else (lambda imgs, key=None, p=None: imgs)
    _loss_G = partial(loss_G, graphdef_G, graphdef_D, augment)
    _loss_D = partial(loss_D, graphdef_G, graphdef_D, augment)
    _loss_D_gp = partial(loss_D_gp, graphdef_G, graphdef_D, augment)
    return (
        make_step(_loss_G, tx_G),      
        make_step(_loss_D, tx_D), 
        make_step_gp(_loss_D_gp, tx_D, gp_micro_batch=gp_micro_batch),   
    )

class R3GANLoss:
    def __init__(self, graphdef_G, graphdef_D, tx_G, tx_D, augment_pipe=None, reg_interval=16,  gp_micro_batch=8):
        self.graphdef_G = graphdef_G
        self.graphdef_D = graphdef_D
        self.augment_pipe = augment_pipe
        self.reg_interval = reg_interval
        self.d_step_count = 0
        self.step_G, self.step_D, self.step_D_gp = build_train_steps(graphdef_G, graphdef_D, tx_G, tx_D, augment_pipe, gp_micro_batch)

    def accumulation_step(self, phase_name, state_G, state_D, opt_state_G, opt_state_D, real_img, real_c, gen_z, cur_gamma, cur_aug_p, key):
        if phase_name == 'G':
            state_G, opt_state_G, loss = self.step_G(state_G, state_D, opt_state_G, real_img, gen_z, real_c, cur_gamma, cur_aug_p, key)
        elif phase_name == 'D':
            if self.d_step_count % self.reg_interval == 0:
                state_D, opt_state_D, loss = self.step_D_gp(
                    state_D, state_G, opt_state_D,
                    real_img, gen_z, real_c,
                    cur_gamma * self.reg_interval, cur_aug_p, key,
                )
            else:
                state_D, opt_state_D, loss = self.step_D(
                    state_D, state_G, opt_state_D,
                    real_img, gen_z, real_c,
                    cur_gamma, cur_aug_p, key,
                )
            self.d_step_count += 1
            
        return state_G, state_D, opt_state_G, opt_state_D, loss