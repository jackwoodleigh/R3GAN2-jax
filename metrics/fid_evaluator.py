"""
JAX-only FID/IS evaluator for R3GAN2 training loop.

Replaces the PyTorch metrics_main.calc_metric path with fid_util's
JAX inception pipeline.

Reference stats are automatically computed and cached on first use,
then loaded from disk on subsequent runs (like metric_utils.py).
"""

import os
import hashlib
import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import nnx
import time
from metrics.fid_util import (
    build_jax_inception,
    compute_stats,
    compute_fid,
    compute_inception_score,
    compute_batch_features,
)
import dnnlib


# ---------------------------------------------------------------------------
# Reference stats: load cached or compute from dataset
# ---------------------------------------------------------------------------

def _cache_path_for_dataset(dataset_kwargs, cache_dir=None):
    """Deterministic cache path from dataset config, like metric_utils.py."""
    md5 = hashlib.md5(repr(sorted(dataset_kwargs.items())).encode("utf-8"))
    tag = f"jax_inception_fid_ref-{md5.hexdigest()}"
    if cache_dir is None:
        cache_dir = dnnlib.make_cache_dir_path("gan-metrics")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, tag + ".npz")


def _compute_reference_stats(dataset_kwargs, inception_net, encoder, batch_size=200, seed=0):
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    num_items = len(dataset)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    is_latent = hasattr(encoder, 'decode') and not isinstance(encoder, StandardRGBEncoder)
    rng = jax.random.PRNGKey(seed)

    all_features = []
    for images, _labels in loader:
        x = jnp.asarray(images.numpy())
        if is_latent:
            rng, k = jax.random.split(rng)
            x = encoder.encode_latents(x, key=k)        # raw latents -> final latents
            x = encoder.decode(x)                        # -> uint8 NCHW numpy
            imgs_np = np.asarray(x).transpose(0, 2, 3, 1)  # NHWC uint8
        else:
            imgs_np = np.asarray(x)
            if imgs_np.ndim == 4 and imgs_np.shape[1] in (1, 3, 4):
                imgs_np = imgs_np.transpose(0, 2, 3, 1)
            if imgs_np.dtype != np.uint8:
                imgs_np = np.clip(imgs_np * 127.5 + 128.0, 0, 255).astype(np.uint8)

        feats = compute_batch_features(imgs_np, inception_net, batch_size)
        all_features.append(np.asarray(feats, dtype=np.float64))

    all_features = np.concatenate(all_features, axis=0)[:num_items]
    return np.mean(all_features, axis=0), np.cov(all_features, rowvar=False)

_INCEPTION_FEATURE_DIM = 2048

def get_or_compute_reference(dataset_kwargs, inception_net, cache_dir=None, batch_size=200):
    """Load cached reference stats or compute from dataset and save."""
    cache_path = _cache_path_for_dataset(dataset_kwargs, cache_dir)
    if jax.process_index() == 0:
        if os.path.isfile(cache_path):
            print(f'  Loading cached ref stats from {cache_path}', flush=True)
            with np.load(cache_path) as data:
                mu = np.array(data["ref_mu"], dtype=np.float64)
                sigma = np.array(data["ref_sigma"], dtype=np.float64)
        else:
            print(f'  Computing reference stats from scratch...', flush=True)
            mu, sigma = _compute_reference_stats(dataset_kwargs, inception_net, batch_size)
            np.savez(cache_path, ref_mu=mu, ref_sigma=sigma)
    else:
        mu = np.zeros(_INCEPTION_FEATURE_DIM, dtype=np.float64)
        sigma = np.zeros((_INCEPTION_FEATURE_DIM, _INCEPTION_FEATURE_DIM), dtype=np.float64)
    
    mu_jax    = jax.experimental.multihost_utils.broadcast_one_to_all(jnp.array(mu))
    sigma_jax = jax.experimental.multihost_utils.broadcast_one_to_all(jnp.array(sigma))
    return {
        "mu":    np.array(mu_jax,    dtype=np.float64),
        "sigma": np.array(sigma_jax, dtype=np.float64),
    }

    #jax.experimental.multihost_utils.sync_global_devices("fid_ref_sync")
    '''with np.load(cache_path) as data:
        result = {"mu": data["ref_mu"], "sigma": data["ref_sigma"]}
    return result'''


# ---------------------------------------------------------------------------
# Sample generation — pmap across all local chips
# ---------------------------------------------------------------------------

def _generate_samples(
    graphdef_G,
    G_state,
    encoder,
    z_dim,
    num_classes,
    num_samples,
    gen_batch_size,
    seed,
):
    """Generate uint8 NHWC numpy images using every chip on every host.

    Each host generates ``ceil(num_samples / num_hosts)`` images.  Within a
    host the work is split evenly across ``jax.local_device_count()`` chips
    via ``jax.pmap``.
    """
    num_hosts = jax.process_count()
    num_local = jax.local_device_count()
    rank = jax.process_index()
    samples_per_host = int(np.ceil(num_samples / num_hosts))

    per_device = gen_batch_size // num_local
    assert per_device > 0, (
        f"gen_batch_size ({gen_batch_size}) must be >= local_device_count ({num_local})"
    )
    total_batch = per_device * num_local

    # Replicate frozen generator state to every local chip.
    replicated_state = jax.device_put_replicated(G_state, jax.local_devices())

    has_classes = num_classes is not None

    # Two entry-points so pmap never sees a None positional arg.
    @jax.pmap
    def _gen_cond(state, z, c, key):
        model = nnx.merge(graphdef_G, state)
        return model(z, c, key=key)

    @jax.pmap
    def _gen_uncond(state, z, key):
        model = nnx.merge(graphdef_G, state)
        return model(z, None, key=key)

    samples_all = []
    rng = jax.random.PRNGKey(seed + rank)
    num_generated = 0


    while num_generated < samples_per_host:
        rng, z_rng, c_rng, g_rng = jax.random.split(rng, 4)

        # [num_local, per_device, z_dim]
        z = jax.random.normal(z_rng, (num_local, per_device, z_dim))
        g_keys = jax.random.split(g_rng, num_local)

        if has_classes:
            c = jax.nn.one_hot(
                jax.random.randint(c_rng, (num_local, per_device), 0, num_classes),
                num_classes,
            )
            raw = _gen_cond(replicated_state, z, c, g_keys)
        else:
            raw = _gen_uncond(replicated_state, z, g_keys)

        # [num_local, per_device, H, W, C] -> [total_batch, H, W, C]
        samples_all.append(jax.device_get(raw).reshape(-1, *raw.shape[2:]))
        num_generated += total_batch
    
    samples = np.concatenate(samples_all, axis=0)[:samples_per_host]
     
    if hasattr(encoder, "decode"):
        decoded = encoder.decode(jnp.asarray(samples, dtype=jnp.float32))
        imgs = np.asarray(decoded).transpose(0, 2, 3, 1)
    else:
        imgs = np.clip(samples * 127.5 + 128.0, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)  
        
    return imgs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_evaluator(dataset_kwargs, inception_batch_size=200, cache_dir=None):
    """Call once before training.  Builds inception net and loads/computes
    reference stats (cached to disk).

    Returns ``(inception_net, stats_ref)``.
    """
    inception_net = build_jax_inception(batch_size=inception_batch_size)
    stats_ref = get_or_compute_reference(
        dataset_kwargs,
        inception_net,
        cache_dir=cache_dir,
        batch_size=inception_batch_size,
    )
    return inception_net, stats_ref


def evaluate(
    ema,
    encoder,
    inception_net,
    stats_ref,
    z_dim,
    num_classes=None,
    num_samples=50000,
    gen_batch_size=256,
    inception_batch_size=200,
    seed=1,
):
    """Generate samples -> JAX Inception -> FID + IS.

    Returns ``(fid, is_mean, is_std)``.
    """
    samples = _generate_samples(
        graphdef_G=ema.graphdef,
        G_state=ema.emas[1],
        encoder=encoder,
        z_dim=z_dim,
        num_classes=num_classes,
        num_samples=num_samples,
        gen_batch_size=gen_batch_size,
        seed=seed,
    )

    stats = compute_stats(
        samples,
        inception_net,
        batch_size=inception_batch_size,
        fid_samples=num_samples,
    )
    fid = compute_fid(
        stats_ref["mu"], stats["mu"],
        stats_ref["sigma"], stats["sigma"],
    )
    if hasattr(encoder, 'unload'):
        encoder.unload()

    return fid