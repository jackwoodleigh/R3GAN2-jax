import jax
from flax import nnx
import jax.numpy as jnp
from jax import lax
from jax import nn as jnn
import math

def Normalize(x, Dimensions=None, ε=1e-4):
    if Dimensions is None:
        Dimensions = tuple(range(1, x.ndim))
    else:
        Dimensions = tuple(Dimensions) if isinstance(Dimensions, list) else Dimensions
    x32 = x.astype(jnp.float32)
    Norm = jnp.sqrt(jnp.sum(x32 ** 2, axis=Dimensions, keepdims=True))
    Norm = ε + Norm * jnp.sqrt(Norm.size / x.size)
    return x / Norm.astype(x.dtype)

def CosineAttention(x, Heads, QKVLayer, ProjectionLayer, Sinks, RoPE, a):
    y = QKVLayer(x)
    y = y.reshape(y.shape[0], Heads, -1, 3, y.shape[2] * y.shape[3])
    y = Normalize(y, Dimensions=2)
    q, k, v = y[..., 0, :], y[..., 1, :], y[..., 2, :]
    
    q = jnp.transpose(q, (0, 1, 3, 2))
    k = jnp.transpose(k, (0, 1, 3, 2))
    #q, k = apply_rope(q, k, RoPE)
    q = jnp.transpose(q, (0, 1, 3, 2))
    k = jnp.transpose(k, (0, 1, 3, 2))
    
    w = jnp.einsum('nhcq,nhck->nhqk', q, a * k / q.shape[2])
    s = a * Sinks.astype(x.dtype).reshape(1, -1, 1, 1) * jnp.ones((w.shape[0], 1, w.shape[2], 1), dtype=x.dtype)
    w = jnp.concatenate([w, s], axis=3)
    w = jax.nn.softmax(w, axis=3)
    
    y = jnp.einsum('nhqk,nhck->nhcq', w[..., :-1], v)
    return ProjectionLayer(y.reshape(x.shape[0], -1, x.shape[2], x.shape[3]))


class LeakyReLU(nnx.Module):
    def __init__(self, α=0.2):        
        self.α = α
        self.Gain = 1 / math.sqrt(((1 + α ** 2) - (1 - α) ** 2 / jnp.pi) / 2)
        
    def __call__(self, x):
        return jnn.leaky_relu(x, negative_slope=self.α) * self.Gain


class BoundedParameter(nnx.Module):
    def __init__(self, Dimension, Bound=1):
        
        self.Value = nnx.Param(jnp.zeros(Dimension))
        self.Bound = Bound
        
    def __call__(self):
        return self.Bound * jnp.tanh(self.Value / self.Bound)

class NormalizedParam(nnx.Variable): pass
class CenteredNormalizedParam(nnx.Variable): pass

class NormalizedWeight(nnx.Module):
    def __init__(self, InputChannels, OutputChannels, Groups, KernelSize, Centered, rngs=None):
        self.Centered = Centered
        ParamType = CenteredNormalizedParam if Centered else NormalizedParam
        self.Weight = ParamType(jax.random.normal(rngs.params(), shape=(OutputChannels, InputChannels // Groups, *KernelSize)))
        
    def Evaluate(self, w):
        if self.Centered:
            w = w - jnp.mean(w, axis=list(range(1, w.ndim)), keepdims=True)
        return Normalize(w)
        
    def __call__(self):
        return self.Evaluate(self.Weight.value.astype(jnp.float32))
    
    def NormalizeWeight(self):
        self.Weight.value = self.Evaluate(self.Weight.value)


class WeightNormalizedConvolution(nnx.Module):
    def __init__(self, InputChannels, OutputChannels, Groups, EnablePadding, KernelSize, Centered, rngs=None):
        
        self.Groups = Groups
        self.EnablePadding = EnablePadding
        self.Weight = NormalizedWeight(InputChannels, OutputChannels, Groups, KernelSize, Centered, rngs=rngs)

    def __call__(self, x, Gain=1):
        w = self.Weight()
        w = w * (Gain / jnp.sqrt(w[0].size))
        w = w.astype(x.dtype)
        if w.ndim == 2:
            return x @ w.T
        
        padding = ((w.shape[-1] // 2, w.shape[-1] // 2),) * 2 if self.EnablePadding else ((0, 0), (0, 0))
    
        return lax.conv_general_dilated(
            x,
            w,
            window_strides=(1, 1),
            padding=padding,
            feature_group_count=self.Groups,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
  

def Convolution(InputChannels, OutputChannels, KernelSize, rngs, Groups=1, Centered=False):
    return WeightNormalizedConvolution(InputChannels, OutputChannels, Groups, True, [KernelSize, KernelSize], Centered, rngs=rngs)

def Linear(InputDimension, OutputDimension, Centered=False,  rngs=None):
    return WeightNormalizedConvolution(InputDimension, OutputDimension, 1, False, [], Centered, rngs=rngs)


class BiasedPointwiseConvolutionWithModulation(nnx.Module):
    def __init__(self, InputChannels, OutputChannels, EmbeddingDimension, Centered=False, rngs=None):
        
        self.Weight = NormalizedWeight(InputChannels + 1, OutputChannels, 1, [1, 1], Centered, rngs=rngs)
        
        if EmbeddingDimension is not None:
            self.EmbeddingLayer = Linear(EmbeddingDimension, InputChannels, Centered, rngs=rngs)
            self.EmbeddingGain = nnx.Param(jnp.zeros([]))
        
    def __call__(self, x, c, Gain=1, key=None):
        w = self.Weight()
        w = w / jnp.sqrt(w[0].size)
        b = w[:, -1, :, :].reshape(-1)
        w = w[:, :-1, :, :] * Gain
        if hasattr(self, 'EmbeddingLayer'):
            c = self.EmbeddingLayer(c, Gain=self.EmbeddingGain) + 1
            x = x * c.reshape(c.shape[0], -1, 1, 1).astype(x.dtype)
 
        return lax.conv_general_dilated(
            x,                          
            w.astype(x.dtype),         
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        ) + b.astype(x.dtype).reshape(1, -1, 1, 1)


class NoisyBiasedPointwiseConvolutionWithModulation(nnx.Module):
    def __init__(self, InputChannels, OutputChannels, EmbeddingDimension, Centered=False, rngs=None):
        
        self.Weight = NormalizedWeight(InputChannels + 2, OutputChannels, 1, [1, 1], Centered, rngs=rngs)
        
        if EmbeddingDimension is not None:
            self.EmbeddingLayer = Linear(EmbeddingDimension, InputChannels, Centered, rngs=rngs)
            self.EmbeddingGain = nnx.Param(jnp.zeros([]))
        
    def __call__(self, x, c, Gain=1, key=None):
        w = self.Weight()
        w = w / jnp.sqrt(w[0].size)
        b = w[:, -1, :, :].reshape(-1)
        s = w[:, -2, :, :].reshape(-1)
        w = w[:, :-2, :, :] * Gain
        if hasattr(self, 'EmbeddingLayer'):
            c = self.EmbeddingLayer(c, Gain=self.EmbeddingGain) + 1
            x = x * c.reshape(c.shape[0], -1, 1, 1).astype(x.dtype)
            
        s = s.reshape(1, -1, 1, 1)
        n = jax.random.normal(key, shape=(x.shape[0], 1, x.shape[2], x.shape[3]))
        
        return lax.conv_general_dilated(
            x,                          
            w.astype(x.dtype),         
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        ) + b.astype(x.dtype).reshape(1, -1, 1, 1) + n * s



class GenerativeBasis(nnx.Module):
    def __init__(self, OutputChannels, rngs=None):
        
        self.Basis = NormalizedWeight(OutputChannels, OutputChannels, OutputChannels // 32, [4, 4], True, rngs=rngs)
        
    def __call__(self, x):
        w = self.Basis()
        x = x.reshape(x.shape[0], -1, 32)
        w = w.reshape(x.shape[1], -1, * w.shape[1:]) / jnp.sqrt(x.shape[-1])
        x = jnp.einsum('ngc,gochw->ngohw', x, w)
        return x.reshape(x.shape[0], -1, * x.shape[3:])
    

class DiscriminativeBasis(nnx.Module):
    def __init__(self, InputChannels, rngs=None):
        self.Basis = WeightNormalizedConvolution(InputChannels, InputChannels, InputChannels // 32, False, [4, 4], True, rngs=rngs)
        
    def __call__(self, x):
        return self.Basis(x).reshape(x.shape[0], -1)



class ClassEmbedder(nnx.Module):
    def __init__(self, NumberOfClasses, EmbeddingDimension, rngs=None):
        self.NumberOfClasses = NumberOfClasses
        self.Weight = NormalizedWeight(EmbeddingDimension, NumberOfClasses, 1, [], True, rngs=rngs)
        self.Weight.Weight.value = jnp.tile(NormalizedWeight(EmbeddingDimension, 1, 1, [], True, rngs=rngs)(), (NumberOfClasses, 1))

    def __call__(self, x):
        return x @ self.Weight().astype(x.dtype)