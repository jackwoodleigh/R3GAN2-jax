import jax
from flax import nnx
import jax.numpy as jnp
from jax import lax
import numpy

def CreateLowpassKernel(Weights, Inplace):
    Kernel = numpy.array([Weights]) if Inplace else numpy.convolve(Weights, [1, 1]).reshape(1, -1)
    Kernel = Kernel.T @ Kernel
    return Kernel / Kernel.sum()

def pixel_shuffle(x, scale):
    N, C, H, W = x.shape
    x = x.reshape(N, C // (scale**2), scale, scale, H, W)
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(N, C // (scale**2), H * scale, W * scale)

def pixel_unshuffle(x, scale):
    N, C, H, W = x.shape
    x = x.reshape(N, C, H // scale, scale, W // scale, scale)
    x = jnp.transpose(x, (0, 1, 3, 5, 2, 4))
    return x.reshape(N, C * scale**2, H // scale, W // scale)


class InterpolativeUpsampler(nnx.Module):
    def __init__(self, Filter):
        self.Kernel = CreateLowpassKernel(Filter, Inplace=False).tolist()
        self.FilterRadius = len(Filter) // 2
        self.PadLow  = self.FilterRadius + 1         
        self.PadHigh = len(Filter) - self.FilterRadius

    def __call__(self, x):
        N, C, H, W = x.shape
        Kernel = jnp.array(self.Kernel)
        Kernel = (4 * Kernel).reshape(1, 1, * Kernel.shape).astype(x.dtype)
        x_flat = x.reshape(N * C, 1, H, W)

        y = lax.conv_transpose(
            x_flat,
            Kernel,
            strides=(2, 2),
            padding=((self.PadLow, self.PadHigh),
                     (self.PadLow, self.PadHigh)),
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )

        return y.reshape(N, C, y.shape[2], y.shape[3])
    
    
class InterpolativeDownsampler(nnx.Module):
    def __init__(self, Filter):
        self.Kernel = CreateLowpassKernel(Filter, Inplace=False).tolist()
        self.FilterRadius = len(Filter) // 2
        
    def __call__(self, x):
        N, C, H, W = x.shape
        Kernel = jnp.array(self.Kernel)
        Kernel = Kernel.reshape(1, 1, * Kernel.shape).astype(x.dtype)
        x_flat = x.reshape(N * C, 1, H, W)
        
        y = lax.conv_general_dilated(
            x_flat,
            Kernel,
            window_strides=(2, 2),
            padding=((self.FilterRadius, self.FilterRadius), (self.FilterRadius, self.FilterRadius)),
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        
        return y.reshape(N, C, y.shape[2], y.shape[3])


class InplaceUpsampler(nnx.Module):
    def __init__(self, Filter):
        self.Kernel = CreateLowpassKernel(Filter, Inplace=True).tolist()
        self.FilterRadius = len(Filter) // 2

    def __call__(self, x):
        x = pixel_shuffle(x, 2)
        N, C, H, W = x.shape
        Kernel = jnp.array(self.Kernel)
        Kernel = Kernel.reshape(1, 1, * Kernel.shape).astype(x.dtype)
        y = lax.conv_general_dilated(
            x.reshape(N * C, 1, H, W),
            Kernel,
            window_strides=(1, 1),
            padding=((self.FilterRadius, self.FilterRadius), (self.FilterRadius, self.FilterRadius)),
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        return y.reshape(N, C, y.shape[2], y.shape[3])


class InplaceDownsampler(nnx.Module):
    def __init__(self, Filter):
        self.Kernel = CreateLowpassKernel(Filter, Inplace=True).tolist()
        self.FilterRadius = len(Filter) // 2

    def __call__(self, x):
        N, C, H, W = x.shape
        Kernel = jnp.array(self.Kernel)
        Kernel = Kernel.reshape(1, 1, * Kernel.shape).astype(x.dtype)
        y = lax.conv_general_dilated(
            x.reshape(N * C, 1, H, W),
            Kernel,
            window_strides=(1, 1),
            padding=((self.FilterRadius, self.FilterRadius), (self.FilterRadius, self.FilterRadius)),
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        return pixel_unshuffle(y.reshape(N, C, y.shape[2], y.shape[3]), 2)