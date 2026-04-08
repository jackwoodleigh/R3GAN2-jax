

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Converting between pixel and latent representations of image data."""

import os
import warnings
import numpy as np
import torch
from utils import persistence
from utils import misc
import jax
import jax.numpy as jnp
import torchax

warnings.filterwarnings('ignore', 'torch.utils._pytree._register_pytree_node is deprecated.')
warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Abstract base class for encoders/decoders that convert back and forth
# between pixel and latent representations of image data.
#
# Logically, "raw pixels" are first encoded into "raw latents" that are
# then further encoded into "final latents". Decoding, on the other hand,
# goes directly from the final latents to raw pixels. The final latents are
# used as inputs and outputs of the model, whereas the raw latents are
# stored in the dataset. This separation provides added flexibility in terms
# of performing just-in-time adjustments, such as data whitening, without
# having to construct a new dataset.
#
# All image data is represented as PyTorch tensors in NCHW order.
# Raw pixels are represented as 3-channel uint8.

@persistence.persistent_class
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x): # raw pixels => final latents
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass

    def encode_latents(self, x): # raw latents => final latents
        raise NotImplementedError # to be overridden by subclass

    def decode(self, x): # final latents => raw pixels
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# Standard RGB encoder that scales the pixel data into [-1, +1].

@persistence.persistent_class
class StandardRGBEncoder(Encoder):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def _to_jax(x, dtype=jnp.float32):
        if isinstance(x, jax.Array):
            return x.astype(dtype)
        if isinstance(x, np.ndarray):
            return jnp.asarray(x, dtype=dtype)
        return jnp.asarray(x.numpy(), dtype=dtype)  

    def encode_pixels(self, x): # raw pixels => raw latents
        return x

    def encode_latents(self, x, key=None, dtype=jnp.float32):
        return self._to_jax(x, dtype=dtype) / 127.5 - 1

    def decode(self, x, dtype=jnp.float32): # final latents => raw pixels
        return (self._to_jax(x, dtype=dtype) * 127.5 + 128).clip(0, 255).astype(jnp.uint8)


@persistence.persistent_class
class Flux2VAEEncoderJAX:
    def __init__(self,
        vae_name            = 'black-forest-labs/FLUX.2-dev',
        latent_stats_path   = 'latent_stats.npz',
        final_mean          = 0,
        final_std           = 0.5,
        batch_size          = 32,
    ):
        self.vae_name = vae_name
        self.batch_size = int(batch_size)
        self._vae = None
        self._env = None
 
        if not os.path.exists(latent_stats_path):
            raise RuntimeError("NO STATS")
 
        stats = np.load(latent_stats_path)
        raw_mean = stats['mean']
        raw_std = stats['std']
        self.scale = jnp.array(np.float32(final_std) / np.float32(raw_std))
        self.bias = jnp.array(np.float32(final_mean) - np.float32(raw_mean) * np.float32(final_std) / np.float32(raw_std))
 
    def __getstate__(self):
        return dict(self.__dict__, _vae=None, _env=None)
 
    # ----- torchax: VAE init -----
 
    def init(self):
        if self._vae is not None:
            return
        # Load weights in pure PyTorch BEFORE torchax intercepts anything
        vae = _load_flux2_vae(self.vae_name)
        # Now enable torchax and transfer
        torchax.enable_globally()
        self._env = torchax.default_env()
        with self._env:
            self._vae = vae.to('jax')
            
    def unload(self):
        """Free VAE weights from device memory."""
        if self._vae is None:
            return
        self._vae = None
        self._env = None
        import gc
        gc.collect()
 
    # ----- cold path: VAE encode/decode -----
 
    def encode_pixels(self, x):  # uint8 jax [B,C,H,W] => raw latents
        
        self.init()
        x = jnp.asarray(x, dtype=jnp.float32) / 127.5 - 1.0
        results = []
        with self._env:
            for i in range(0, x.shape[0], self.batch_size):
                batch = torchax.tensor.Tensor(x[i:i + self.batch_size], self._env)
                d = self._vae.encode(batch)['latent_dist']
                out = torch.cat([d.mean, d.std], dim=1)
                results.append(out.jax())
        return jnp.concatenate(results, axis=0)
 
    def decode(self, x):  # final latents => uint8 jax [B,C,H,W]
        self.init()
        x = x.astype(jnp.float32)
        x = (x - self.bias.reshape(1, -1, 1, 1)) / self.scale.reshape(1, -1, 1, 1)
        results = []
        with self._env:
            for i in range(0, x.shape[0], self.batch_size):
                batch = torchax.tensor.Tensor(x[i:i + self.batch_size], self._env)
                out = self._vae.decode(batch)['sample']
                results.append(out.jax())
        x = jnp.concatenate(results, axis=0)
        return ((x / 2 + 0.5).clip(0, 1) * 255).astype(jnp.uint8)
 

    @staticmethod
    @jax.jit
    def _encode_latents_impl(x, key, scale, bias):
        x = x.astype(jnp.float32)
        C = x.shape[1] // 2
        mean, std = x[:, :C], x[:, C:]
        x = mean + jax.random.normal(key, mean.shape, dtype=jnp.float32) * std
        x = x * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)
        return x
 
    def encode_latents(self, x, key):  # raw latents [B,2C,H,W] => final latents [B,C,H,W]
        return self._encode_latents_impl(x, key, self.scale, self.bias)
 
 
# ---------------------------------------------------------------------------
 
def _load_flux2_vae(vae_name='black-forest-labs/FLUX.2-dev'):
    import torch
    import dnnlib
    cache_dir = dnnlib.make_cache_dir_path('diffusers')
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HOME'] = cache_dir
    token = os.environ.get("HF_TOKEN")
 
    import diffusers
    try:
        vae = diffusers.models.AutoencoderKLFlux2.from_pretrained(
            vae_name, subfolder="vae", cache_dir=cache_dir, local_files_only=True)
    except Exception:
        vae = diffusers.models.AutoencoderKLFlux2.from_pretrained(
            vae_name, subfolder="vae", cache_dir=cache_dir, token=token)
    return vae.eval().requires_grad_(False)
