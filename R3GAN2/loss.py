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


def loss_D_full(graphdef_G, graphdef_D, augment, reg_interval, state_D, state_G, RealSamples, z, c, cur_gamma, cur_aug_p, key, reg):
    key = jax.random.fold_in(key, ord('D'))
    G = nnx.merge(graphdef_G, state_G)
    D = nnx.merge(graphdef_D, state_D)
    FakeSamples = lax.stop_gradient(G(z, c, key))

    def joint_forward(fake, real):
        TransformedFake, TransformedReal = augment([fake.astype(jnp.float32), real.astype(jnp.float32)], cur_aug_p, key)
        return D(TransformedFake, c), D(TransformedReal, c)
    
    (FakeLogits, RealLogits), vjp_fn = jax.vjp(joint_forward, FakeSamples, RealSamples)
    AdversarialLoss = jnn.softplus(-(RealLogits - FakeLogits))
    R1Penalty, R2Penalty = FusedZeroCenteredGradientPenalty(vjp_fn, FakeLogits, RealLogits)
    return (AdversarialLoss + (cur_gamma / 2) * (R1Penalty + R2Penalty)).mean() 


def loss_D_single_reg(graphdef_G, graphdef_D, augment, reg_interval, state_D, state_G, RealSamples, z, c, cur_gamma, cur_aug_p, key, reg):
    key = jax.random.fold_in(key, ord('D'))
    G = nnx.merge(graphdef_G, state_G)
    D = nnx.merge(graphdef_D, state_D)
    FakeSamples = lax.stop_gradient(G(z, c, key))

    def D_fwd_real(real):
        TransformedReal, = augment([real.astype(jnp.float32)], cur_aug_p, key)
        return D(TransformedReal, c)
    def D_fwd_fake(fake):
        TransformedFake, = augment([fake.astype(jnp.float32)], cur_aug_p, key)
        return D(TransformedFake, c)
    
    RealLogits, vjp_real = jax.vjp(D_fwd_real, RealSamples)
    FakeLogits, vjp_fake = jax.vjp(D_fwd_fake, FakeSamples)
    AdversarialLoss = jnn.softplus(-(RealLogits - FakeLogits))

    def with_r1(_):
        R1grads = vjp_real(jnp.ones_like(RealLogits))[0]
        R1Penalty = (R1grads ** 2).sum(axis=(1, 2, 3))
        return AdversarialLoss.mean() + ((cur_gamma / 2) * R1Penalty).mean() * reg_interval 

    def with_r2(_):
        R2grads = vjp_fake(jnp.ones_like(FakeLogits))[0]
        R2Penalty = (R2grads ** 2).sum(axis=(1, 2, 3))
        return AdversarialLoss.mean() + ((cur_gamma / 2) * R2Penalty).mean() * reg_interval 

    return lax.switch(reg - 1, [with_r1, with_r2], operand=None)

def loss_D_r2(graphdef_G, graphdef_D, augment, reg_interval, state_D, state_G, RealSamples, z, c, cur_gamma, cur_aug_p, key, reg):
    key = jax.random.fold_in(key, ord('D'))
    G = nnx.merge(graphdef_G, state_G)
    D = nnx.merge(graphdef_D, state_D)
    FakeSamples = lax.stop_gradient(G(z, c, key))
    def D_fwd_real(real):
        TransformedReal, = augment([real.astype(jnp.float32)], cur_aug_p, key)
        return D(TransformedReal, c)
    def D_fwd_fake(fake):
        TransformedFake, = augment([fake.astype(jnp.float32)], cur_aug_p, key)
        return D(TransformedFake, c)
    
    RealLogits = D_fwd_real(RealSamples)
    FakeLogits, vjp_fake = jax.vjp(D_fwd_fake, FakeSamples)
    AdversarialLoss = jnn.softplus(-(RealLogits - FakeLogits))
    R2grads = vjp_fake(jnp.ones_like(FakeLogits))[0]
    R2Penalty = (R2grads ** 2).sum(axis=(1, 2, 3))
    return AdversarialLoss.mean() + ((cur_gamma / 2) * R2Penalty).mean() * reg_interval

def loss_D_r1(graphdef_G, graphdef_D, augment, reg_interval, state_D, state_G, RealSamples, z, c, cur_gamma, cur_aug_p, key, reg):
    key = jax.random.fold_in(key, ord('D'))
    G = nnx.merge(graphdef_G, state_G)
    D = nnx.merge(graphdef_D, state_D)
    FakeSamples = lax.stop_gradient(G(z, c, key))
    def D_fwd_real(real):
        TransformedReal, = augment([real.astype(jnp.float32)], cur_aug_p, key)
        return D(TransformedReal, c)
    def D_fwd_fake(fake):
        TransformedFake, = augment([fake.astype(jnp.float32)], cur_aug_p, key)
        return D(TransformedFake, c)
    
    RealLogits, vjp_real = jax.vjp(D_fwd_real, RealSamples)
    FakeLogits = D_fwd_fake(FakeSamples)
    AdversarialLoss = jnn.softplus(-(RealLogits - FakeLogits))
    R1grads = vjp_real(jnp.ones_like(RealLogits))[0]
    R1Penalty = (R1grads ** 2).sum(axis=(1, 2, 3))
    return AdversarialLoss.mean() + ((cur_gamma / 2) * R1Penalty).mean() * reg_interval

def r2(graphdef_G, graphdef_D, augment, reg_interval, state_D, state_G, RealSamples, z, c, cur_gamma, cur_aug_p, key, reg):
    key = jax.random.fold_in(key, ord('D'))
    G = nnx.merge(graphdef_G, state_G)
    D = nnx.merge(graphdef_D, state_D)
    FakeSamples = lax.stop_gradient(G(z, c, key))
    def D_fwd_fake(fake):
            TransformedFake, = augment([fake.astype(jnp.float32)], cur_aug_p, key)
            return D(TransformedFake, c)
    FakeLogits, vjp_fake = jax.vjp(D_fwd_fake, FakeSamples)
    R2grads = vjp_fake(jnp.ones_like(FakeLogits))[0]
    R2Penalty = (R2grads ** 2).sum(axis=(1, 2, 3))
    return ((cur_gamma / 2) * R2Penalty).mean() * reg_interval

def r1(graphdef_G, graphdef_D, augment, reg_interval, state_D, state_G, RealSamples, z, c, cur_gamma, cur_aug_p, key, reg):
    key = jax.random.fold_in(key, ord('D'))
    D = nnx.merge(graphdef_D, state_D)
    def D_fwd_real(real):
        TransformedReal, = augment([real.astype(jnp.float32)], cur_aug_p, key)
        return D(TransformedReal, c)
    
    RealLogits, vjp_real = jax.vjp(D_fwd_real, RealSamples)
    R1grads = vjp_real(jnp.ones_like(RealLogits))[0]
    R1Penalty = (R1grads ** 2).sum(axis=(1, 2, 3))

    return ((cur_gamma / 2) * R1Penalty).mean() * reg_interval

def loss_D_noreg(graphdef_G, graphdef_D, augment, state_D, state_G, RealSamples, z, c, cur_gamma, cur_aug_p, key, reg):
    key = jax.random.fold_in(key, ord('D'))
    G = nnx.merge(graphdef_G, state_G)
    D = nnx.merge(graphdef_D, state_D)
    FakeSamples = lax.stop_gradient(G(z, c, key))
    TransformedFake, TransformedReal = augment([FakeSamples.astype(jnp.float32), RealSamples.astype(jnp.float32)], cur_aug_p, key)
    FakeLogits, RealLogits = D(TransformedFake, c), D(TransformedReal, c)
    return jnn.softplus(-(RealLogits - FakeLogits)).mean()

def loss_G(graphdef_G, graphdef_D, augment, state_G, state_D, RealSamples, z, c, cur_gamma, cur_aug_p, key, reg):
    key = jax.random.fold_in(key, ord('G'))
    G = nnx.merge(graphdef_G, state_G)
    D = nnx.merge(graphdef_D, state_D)
    FakeSamples = G(z, c, key)
    TransformedFake, TransformedReal = augment([FakeSamples.astype(jnp.float32), RealSamples.astype(jnp.float32)], cur_aug_p, key)
    FakeLogits, RealLogits = D(TransformedFake, c), D(TransformedReal, c)
    RelativisticLogits = FakeLogits - RealLogits
    return jnn.softplus(-RelativisticLogits).mean()

def make_step(loss_fn, tx):
    @partial(jax.pmap, axis_name='batch')
    def step(state, state_other, opt_state, real_imgs, noises, conditions, cur_gamma, cur_aug_p, key, reg):
        indices = jnp.arange(real_imgs.shape[0])

        # Grad accumulation fn
        def scan_fn(acc_grads, chunk):
            real_img, z, c, idx = chunk
            chunk_key = jax.random.fold_in(key, idx)
            loss, grads = jax.value_and_grad(loss_fn)(state, state_other, real_img, z, c, cur_gamma, cur_aug_p, chunk_key, reg)
            return jax.tree.map(lambda a, b: a + b, acc_grads, grads), loss

        init_grads = jax.tree.map(jnp.zeros_like, state)
        acc_grads, losses = jax.lax.scan(scan_fn, init_grads, (real_imgs, noises, conditions, indices))
        acc_grads = jax.tree.map(lambda g: g / real_imgs.shape[0], acc_grads)
        acc_grads = lax.pmean(acc_grads, axis_name='batch')
        updates, new_opt_state = tx.update(acc_grads, opt_state)
        
        # Weight normalization 
        return normalize_state_weights(optax.apply_updates(state, updates)), new_opt_state, jnp.mean(losses)
    return step

def build_train_steps(graphdef_G, graphdef_D, tx_G, tx_D, augment_pipe=None, reg_interval=None):
    augment = augment_pipe if augment_pipe is not None else (lambda imgs, key=None, p=None: imgs)
    _loss_G = partial(loss_G, graphdef_G, graphdef_D, augment)
    _loss_D_noreg = partial(loss_D_noreg, graphdef_G, graphdef_D, augment)
    _loss_r1 = partial(r1, graphdef_G, graphdef_D, augment, reg_interval)
    _loss_r2 = partial(r2, graphdef_G, graphdef_D, augment, reg_interval)
    
    return make_step(_loss_G, tx_G), make_step(_loss_D_noreg, tx_D), make_step(_loss_r1, tx_D), make_step(_loss_r2, tx_D)

class R3GANLoss:
    def __init__(self, graphdef_G, graphdef_D, tx_G, tx_D, augment_pipe=None, reg_interval=None):
        self.graphdef_G = graphdef_G
        self.graphdef_D = graphdef_D
        self.augment_pipe = augment_pipe
        self.step_G, self.step_D_noreg, self.step_r1, self.step_r2   = build_train_steps(graphdef_G, graphdef_D, tx_G, tx_D, augment_pipe, reg_interval)

    def accumulation_step(self, phase_name, state_G, state_D, opt_state_G, opt_state_D, real_img, real_c, gen_z, cur_gamma, cur_aug_p, key, reg):
        if phase_name == 'G':
            state_G, opt_state_G, loss = self.step_G(state_G, state_D, opt_state_G, real_img, gen_z, real_c, cur_gamma, cur_aug_p, key, reg)
        elif phase_name == 'D':
            if int(reg[0]) == 0:
                state_D, opt_state_D, loss = self.step_D_noreg(state_D, state_G, opt_state_D, real_img, gen_z, real_c, cur_gamma, cur_aug_p, key, reg)
            elif int(reg[0]) == 1:
                state_D, opt_state_D, loss = self.step_r1(state_D, state_G, opt_state_D, real_img, gen_z, real_c, cur_gamma, cur_aug_p, key, reg)
            elif int(reg[0]) == 2:
                state_D, opt_state_D, loss = self.step_r2(state_D, state_G, opt_state_D, real_img, gen_z, real_c, cur_gamma, cur_aug_p, key, reg)
       
        return state_G, state_D, opt_state_G, opt_state_D, loss