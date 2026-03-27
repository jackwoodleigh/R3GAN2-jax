import jax
import torch
import numpy as np
from torch.utils.data import DataLoader
from jax import lax, random
import jax.numpy as jnp
from flax import nnx
from torch.utils.data.distributed import DistributedSampler
import training.encoders as encoders
from functools import partial
import optax
import time
import os
import psutil
from torch.utils.tensorboard import SummaryWriter
import sys
import dnnlib
 
from R3GAN2.Network import Generator, Discriminator
from utils.misc import InfiniteSampler
from R3GAN2.loss import R3GANLoss
from training.augment import AugmentPipe
from training.phema import PowerFunctionEMA
#from metrics import metrics_main

from metrics.jax_fid_evaluator import build_evaluator, evaluate


def worker_init_fn(worker_id):
    dataset = torch.utils.data.get_worker_info().dataset
    dataset._zipfile = None  

def format_time(seconds):
    s = int(seconds)
    if s < 60:    return f"{s}s"
    if s < 3600:  return f"{s//60}m {s%60:02d}s"
    return f"{s//3600}h {(s%3600)//60:02d}m {s%60:02d}s"

def shard(x):
    n = jax.local_device_count()
    return x.reshape((n, x.shape[0] // n) + x.shape[1:])

# [local_B, ...] -> [local_devices, num_chunks, chunk_size, ...]
def shard_with_chunks(x, chunk_size):
    n_dev = jax.local_device_count()
    local_B = x.shape[0]
    n_chunks = local_B // (n_dev * chunk_size)
    return x.reshape((n_dev, n_chunks, chunk_size) + x.shape[1:])

def to_jax(x, dtype=jnp.float32, device=None, nhwc=False):
    arr = jnp.asarray(x.numpy(), dtype=dtype)
    if nhwc:
        arr = jnp.transpose(arr, (0, 2, 3, 1))
    if device is not None:
        arr = jax.device_put(arr, device)
    return arr
    
def cosine_decay_with_warmup(cur_nimg, base_value, total_nimg, final_value=0.0, warmup_value=0.0, warmup_nimg=0, hold_base_value_nimg=0, post_cosine_decay_ref_nimg=None):
    decay = 0.5 * (1 + np.cos(np.pi * (cur_nimg - warmup_nimg - hold_base_value_nimg) / float(total_nimg - warmup_nimg - hold_base_value_nimg)))
    cur_value = base_value + (1 - decay) * (final_value - base_value)
    if hold_base_value_nimg > 0:
        cur_value = np.where(cur_nimg > warmup_nimg + hold_base_value_nimg, cur_value, base_value)
    if warmup_nimg > 0:
        slope = (base_value - warmup_value) / warmup_nimg
        warmup_v = slope * cur_nimg + warmup_value
        cur_value = np.where(cur_nimg < warmup_nimg, warmup_v, cur_value)
    if post_cosine_decay_ref_nimg is not None:
        final_value /= np.sqrt(max((cur_nimg + post_cosine_decay_ref_nimg - total_nimg - warmup_nimg - hold_base_value_nimg) / post_cosine_decay_ref_nimg, 1))
    return float(np.where(cur_nimg > total_nimg, final_value, cur_value))

def edm2_learning_rate_schedule(cur_nimg, batch_size, ref_lr, ref_batches, rampup_Mimg):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

def schedulers(lr_scheduler, beta2_scheduler, gamma_scheduler, aug_scheduler, cur_nimg):
    cur_lr = edm2_learning_rate_schedule(cur_nimg, **lr_scheduler)
    cur_beta2 = cosine_decay_with_warmup(cur_nimg, **beta2_scheduler)
    cur_gamma = cosine_decay_with_warmup(cur_nimg, **gamma_scheduler)
    cur_aug_p = cosine_decay_with_warmup(cur_nimg, **aug_scheduler)
    cur_gamma  = jnp.full((jax.local_device_count(),), cur_gamma)
    cur_aug_p  = jnp.full((jax.local_device_count(),), cur_aug_p)
    return cur_lr, cur_beta2, cur_gamma, cur_aug_p


def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    eval_set_kwargs         = {},       # Options for eval set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    lr_scheduler            = None,
    beta2_scheduler         = None,
    ema_kwargs              = None,
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    gamma_scheduler         = None,
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_devices             = 1,        # Number of GPUs/devices participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    g_batch_gpu             = 4,        # Number of samples processed at a time by one GPU.
    d_batch_gpu             = 4,        # Number of samples processed at a time by one GPU.
    aug_scheduler           = None,
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    ema_snapshot_ticks      = 50,
    resume_pkl              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    start_time = time.time()
    rank = jax.process_index()
    num_hosts = jax.process_count()
    local_devices = jax.local_device_count()
    local_batch_size = batch_size // num_hosts
    
    if g_batch_gpu is None:
        g_batch_gpu = local_batch_size // jax.local_device_count()
    if d_batch_gpu is None:
        d_batch_gpu = local_batch_size // jax.local_device_count()
        
    assert local_batch_size % (local_devices * g_batch_gpu) == 0, f"local_batch_size {local_batch_size} not divisible by {local_devices} * {g_batch_gpu}"

    z_dim = G_kwargs['NoiseDimension']
    rngs = random.key(64 + rank)
    
    # Data
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = InfiniteSampler(dataset=training_set, rank=rank, num_replicas=jax.process_count(), seed=random_seed)
    dataloader = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, worker_init_fn=worker_init_fn, batch_size=local_batch_size, **data_loader_kwargs))
    
    # Model 
    G = Generator(**G_kwargs, rngs=nnx.Rngs(42))
    D = Discriminator(**D_kwargs, rngs=nnx.Rngs(43))        
    graphdef_G, state_G = nnx.split(G)
    graphdef_D, state_D = nnx.split(D)
    
    # EMA
    ema = PowerFunctionEMA(net=G, **ema_kwargs)
    ema_preview = ema.emas[1]

    # Optimizer 
    tx_G = optax.inject_hyperparams(optax.adam)(**G_opt_kwargs)
    tx_D = optax.inject_hyperparams(optax.adam)(**D_opt_kwargs)
    opt_state_G = tx_G.init(state_G)
    opt_state_D = tx_D.init(state_D)

    # Replicating States
    state_G     = jax.device_put_replicated(state_G,     jax.local_devices())
    state_D     = jax.device_put_replicated(state_D,     jax.local_devices())
    opt_state_G = jax.device_put_replicated(opt_state_G, jax.local_devices())
    opt_state_D = jax.device_put_replicated(opt_state_D, jax.local_devices())

    # Augmentation 
    augment_pipe = None
    if (augment_kwargs is not None) and (aug_scheduler is not None):
        augment_pipe = AugmentPipe(**augment_kwargs)
    
    # Loss
    loss_pipe = R3GANLoss(graphdef_G, graphdef_D, tx_G, tx_D, augment_pipe)
    
    # Schedulers
    _schedulers = partial(schedulers, lr_scheduler, beta2_scheduler, gamma_scheduler, aug_scheduler)
    
    inception_net = None
    stats_ref = None
    
    if training_set.num_channels == 3:
        encoder = encoders.StandardRGBEncoder()
    elif training_set.num_channels == 8:
        encoder = encoders.StabilityVAEEncoder()
    else:
        encoder = encoders.Flux2VAEEncoder()
    
    # Model info
    if rank == 0:
        G_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(G)))
        D_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(D)))
        print(f"  G params:            {G_params:,}", flush=True)
        print(f"  D params:            {D_params:,}", flush=True)
        print(f"  Encoder:             {encoder.__class__.__name__}", flush=True)
        print("=" * 60, flush=True)
        print(flush=True)

    # Trackers
    cur_tick = 0
    cur_nimg = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    phase = ['D', 'G']

    writer = SummaryWriter(log_dir=run_dir) if rank == 0 else None
    if rank == 0:
        print("Training...", flush=True)
        
    tick_start_time = time.time()
    while True:
        # Random Keys
        D_key = random.fold_in(random.fold_in(rngs, 0), cur_nimg)
        G_key = random.fold_in(random.fold_in(rngs, 1), cur_nimg)
        z_D_key = random.fold_in(random.fold_in(rngs, 2), cur_nimg)
        z_G_key = random.fold_in(random.fold_in(rngs, 3), cur_nimg)
        
        all_keys = [
            random.split(D_key, jax.local_device_count()),
            random.split(G_key, jax.local_device_count()),
        ]
        
        # Dataloading
        D_img, D_img_c = next(dataloader)
        D_img = encoder.encode_latents(D_img)
        D_z = random.normal(z_D_key, shape=(local_batch_size, z_dim))

        G_img, G_img_c = next(dataloader)
        G_img = encoder.encode_latents(G_img)
        G_z = random.normal(z_G_key, shape=(local_batch_size, z_dim))

        all_real_img = [
            shard_with_chunks(to_jax(D_img), d_batch_gpu),
            shard_with_chunks(to_jax(G_img), g_batch_gpu),
        ]
        all_real_c = [
            shard_with_chunks(to_jax(D_img_c, dtype=jnp.float32), d_batch_gpu),
            shard_with_chunks(to_jax(G_img_c, dtype=jnp.float32), g_batch_gpu),
        ]
        all_gen_z = [
            shard_with_chunks(D_z, d_batch_gpu),
            shard_with_chunks(G_z, g_batch_gpu),
        ]
        
        # Updating Schedulers 
        cur_lr, cur_beta2, cur_gamma, cur_aug_p = _schedulers(cur_nimg)
        opt_state_G.hyperparams['learning_rate'] = jnp.full_like(opt_state_G.hyperparams['learning_rate'], cur_lr)
        opt_state_G.hyperparams['b2']            = jnp.full_like(opt_state_G.hyperparams['b2'], cur_beta2)
        opt_state_D.hyperparams['learning_rate'] = jnp.full_like(opt_state_D.hyperparams['learning_rate'], cur_lr)
        opt_state_D.hyperparams['b2']            = jnp.full_like(opt_state_D.hyperparams['b2'], cur_beta2)
        

        losses = {}
        for phase_name, phase_real_img, phase_real_c, phase_gen_z, phase_key in zip(phase, all_real_img, all_real_c, all_gen_z, all_keys):
            info = loss_pipe.accumulation_step(phase_name, state_G, state_D, opt_state_G, opt_state_D, phase_real_img, phase_real_c, phase_gen_z, cur_gamma, cur_aug_p, phase_key)
            state_G, state_D, opt_state_G, opt_state_D, loss_val = info
            losses[phase_name] = loss_val
       
        cur_nimg += batch_size
        batch_idx += 1
        
        
        # Update EMA
        net_state = jax.tree_util.tree_map(lambda x: x[0], state_G)
        ema.update(net_state, cur_nimg, batch_size)
        
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        
        # Printing tick info
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {cur_tick:<5d}"]
        fields += [f"kimg {cur_nimg / 1e3:<8.1f}"]
        fields += [f"time {format_time(tick_end_time - start_time):<12s}"]
        fields += [f"sec/tick {tick_end_time - tick_start_time:<7.1f}"]
        fields += [f"sec/kimg {(tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3:<7.2f}"]
        fields += [f"maintenance {maintenance_time:<6.1f}"]
        if rank == 0:
            print(" | ".join(fields), flush=True)
        
        # Logging to tensorboard 
        if writer is not None:
            kimg = cur_nimg / 1e3
            for k, v in losses.items():
                writer.add_scalar(f'Loss/{k}', float(jax.device_get(v[0])), cur_nimg)
            writer.add_scalar('Schedule/lr', cur_lr, cur_nimg)
            writer.add_scalar('Schedule/beta2', cur_beta2, cur_nimg)
            writer.add_scalar('Schedule/gamma', float(cur_gamma[0]), cur_nimg)
            writer.add_scalar('Schedule/aug_p', float(cur_aug_p[0]), cur_nimg)
            writer.add_scalar('Perf/sec_per_tick', tick_end_time - tick_start_time, cur_nimg)
            writer.add_scalar('Perf/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3, cur_nimg)
            writer.flush()

        # Evaluate metrics. len(metrics) > 0 and
        if cur_tick % network_snapshot_ticks == 0:
            if inception_net is None:
                inception_net, stats_ref = build_evaluator(eval_set_kwargs)
            fid, is_mean, is_std = evaluate(
                ema, encoder, inception_net, stats_ref,
                z_dim=z_dim,
                num_classes=G_kwargs.get('NumberOfClasses'),
                seed=cur_nimg
            )
            if rank == 0:
                print(f"  FID: {fid:.4f} | IS: {is_mean:.4f} +/- {is_std:.4f}", flush=True)
            if writer is not None:
                writer.add_scalar('Metrics/FID', fid, cur_nimg)
                writer.add_scalar('Metrics/IS', is_mean, cur_nimg)
                writer.add_scalar('Metrics/IS_std', is_std, cur_nimg)
            jax.experimental.multihost_utils.sync_global_devices("metrics_sync")
            
        '''if len(metrics) > 0 and cur_tick % network_snapshot_ticks == 0:
            encoder_kwargs = dnnlib.EasyDict(class_name=f'training.encoders.{encoder.__class__.__name__}')
            print('Evaluating metrics...', flush=True)
            for metric in metrics:
                result_dict = metrics_main.calc_metric(
                    metric=metric,
                    encoder_kwargs=encoder_kwargs,
                    dataset_kwargs=eval_set_kwargs,
                    graphdef_G=ema.graphdef,
                    G_state=ema.emas[1],
                    num_devices=1,
                    rank=0,
                    device=torch.device('cpu'),
                    seed=cur_nimg,
                )
            if rank == 0:
                metrics_main.report_metric(result_dict, run_dir=run_dir)
                if writer is not None:
                    for k, v in result_dict.results.items():
                        writer.add_scalar(f'Metrics/{k}', v, cur_nimg)
            
            jax.experimental.multihost_utils.sync_global_devices("metrics_sync")'''
            
            
        cur_tick += 1
        tick_start_time = tick_end_time
        tick_start_nimg = cur_nimg
        maintenance_time = time.time() - tick_end_time
        

        
        
