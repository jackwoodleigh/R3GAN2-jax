import jax
import jax.numpy as jnp
from jax import lax


def CollectGeneratorFeatures(G, z, y, key):
    x = jnp.concatenate([z, G.EmbeddingLayer(y)], axis=1) if hasattr(G, 'EmbeddingLayer') else z
    x = G.Head(x).astype(jnp.bfloat16)
    f = []
    for i, (Layer, Transition) in enumerate(zip(G.MainLayers[:-1], G.TransitionLayers)):
        x, acc_var = Layer(x, None, key=jax.random.fold_in(key, i))
        f.append(x * lax.rsqrt(acc_var).reshape(1, -1, 1, 1).astype(x.dtype))
        x = Transition(x, Gain=lax.rsqrt(acc_var))
    x, acc_var = G.MainLayers[-1](x, None, key=jax.random.fold_in(key, len(G.MainLayers) - 1))
    f.append(x * lax.rsqrt(acc_var).reshape(1, -1, 1, 1).astype(x.dtype))
    return f


def CollectDiscriminatorFeatures(D, x, y):
    x = D.ExtractionLayer(x.astype(jnp.bfloat16))
    f = []
    for Layer, Transition in zip(D.MainLayers[:-1], D.TransitionLayers):
        x, acc_var = Layer(x, None)
        f.append(x * lax.rsqrt(acc_var).reshape(1, -1, 1, 1).astype(x.dtype))
        x = Transition(x, Gain=lax.rsqrt(acc_var))
    x, acc_var = D.MainLayers[-1](x, None)
    f.append(x * lax.rsqrt(acc_var).reshape(1, -1, 1, 1).astype(x.dtype))
    return f


def CollectMagnitude(x, mode='avg'):
    N, C = x.shape[0], x.shape[1]
    x = x.reshape(N, C, -1).astype(jnp.float32)
    mag = jnp.sqrt((x ** 2).sum(axis=2) / x.shape[2])
    mag = mag.mean(axis=1) if mode == 'avg' else mag.max(axis=1)
    return float(mag.mean(axis=0))
