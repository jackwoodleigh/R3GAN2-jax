# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import click
import re
import json
import tempfile
import torch

import jax
jax.distributed.initialize()

import dnnlib
from training import training_loop

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of Devices:      {c.num_devices}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--eval',         help='Evaluation data', metavar='[ZIP|DIR]',                    type=str, default='none', show_default=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--preset',       help='Preset configs', metavar='STR',                           type=str, required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Enable Augmentation', metavar='BOOL',                     type=bool, default=True, show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)

# Misc hyperparameters.
@click.option('--g-batch-gpu',  help='Limit batch size per GPU for G', metavar='INT',           type=click.IntRange(min=1))
@click.option('--d-batch-gpu',  help='Limit batch size per GPU for D', metavar='INT',           type=click.IntRange(min=1))

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=10000000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--ema-snap',     help='How often to save ema snapshots', metavar='TICKS',        type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=8, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)
@click.option('--tpu-id',       help='TPU ID', metavar='STR',            type=str, default=None, show_default=True)

def main(**kwargs):
    rank = jax.process_index()
    
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    
    c.G_kwargs = dnnlib.EasyDict()
    c.D_kwargs = dnnlib.EasyDict()
    
    c.G_opt_kwargs = dnnlib.EasyDict(learning_rate=0.0, b1=0., b2=0.9, eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(learning_rate=0.0, b1=0., b2=0.9, eps=1e-8)

    c.loss_kwargs = dnnlib.EasyDict()
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=False, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    if opts.eval == 'none':
        opts.eval = opts.data
    c.eval_set_kwargs, _ = init_dataset_kwargs(data=opts.eval)

    # Hyperparameters & settings.
    c.num_devices = opts.gpus
    c.batch_size = opts.batch
    c.g_batch_gpu = opts.g_batch_gpu 
    c.d_batch_gpu = opts.d_batch_gpu 
    
    if opts.preset == 'CIFAR10':
        WidthPerStage = [x // 2 for x in [1024, 1024, 1024]]
        BlocksPerStage = [['FFN', 'FFN', 'FFN', 'FFN'], ['FFN', 'FFN', 'FFN', 'FFN'], ['FFN', 'FFN', 'FFN', 'FFN']]
        NoiseDimension = 64
        aug_config = dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=0.5, contrast=0.5, lumaflip=0.5, hue=0.5, saturation=0.5, cutout=1)
        ema_stds = [0.010, 0.050, 0.100]
       
        c.G_kwargs.ClassEmbeddingDimension = NoiseDimension
        c.D_kwargs.ClassEmbeddingDimension = WidthPerStage[0]
       
        decay_nimg = 2e7
       
        c.aug_scheduler = { 'base_value': 0, 'final_value': 0.55, 'total_nimg': decay_nimg }
        c.lr_scheduler = { 'batch_size': 512, 'ref_lr': 100e-4, 'ref_batches': 1e3, 'rampup_Mimg': 1 }
        c.gamma_scheduler = { 'base_value': 0.05, 'final_value': 0.005, 'total_nimg': decay_nimg }
        c.beta2_scheduler = { 'base_value': 0.9, 'final_value': 0.99, 'total_nimg': decay_nimg }

    if opts.preset == 'ImageNet-Ablation':
        WidthPerStage = [x // 2 for x in [1024, 1024, 1024]]
        BlocksPerStage = [['FFN', 'FFN', 'FFN', 'FFN'], ['FFN', 'FFN', 'FFN', 'FFN'], ['FFN', 'FFN', 'FFN', 'FFN']]
        NoiseDimension = 64
        aug_config = dict(rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, cutout=1)
        ema_stds = [0.050, 0.100, 0.200, 0.300]
       
        c.G_kwargs.ClassEmbeddingDimension = NoiseDimension
        c.D_kwargs.ClassEmbeddingDimension = WidthPerStage[0]
       
        decay_nimg = 2e8 / 2
       
        c.aug_scheduler = { 'base_value': 0, 'final_value': 0.3, 'total_nimg': decay_nimg }
        c.lr_scheduler = { 'batch_size': 4096, 'ref_lr': 50e-4, 'ref_batches': 70000, 'rampup_Mimg': 10 }
        c.gamma_scheduler = { 'base_value': 5, 'final_value': 0.5, 'total_nimg': decay_nimg }
        c.beta2_scheduler = { 'base_value': 0.9, 'final_value': 0.9, 'total_nimg': decay_nimg }

    c.G_kwargs.NoiseDimension = NoiseDimension
    c.G_kwargs.WidthPerStage = WidthPerStage
    c.G_kwargs.BlocksPerStage = BlocksPerStage
    c.G_kwargs.FFNWidthRatio = 2
    c.G_kwargs.ChannelsPerConvolutionGroup = 32
    
    c.D_kwargs.WidthPerStage = [*reversed(WidthPerStage)]
    c.D_kwargs.BlocksPerStage = [*reversed(BlocksPerStage)]
    c.D_kwargs.FFNWidthRatio = 2
    c.D_kwargs.ChannelsPerConvolutionGroup = 32
    
    
    # TEMP
    c.G_kwargs.ModulationDimension = WidthPerStage[0]
    c.G_kwargs.OutputChannels = 3
    c.G_kwargs.MLPWidthRatio = 2
    c.G_kwargs.AttentionWidthRatio = 1
    c.G_kwargs.ChannelsPerAttentionHead = 64
    c.G_kwargs.NumberOfClasses = 10  # for CIFAR10 conditional
    
    c.D_kwargs.ModulationDimension = WidthPerStage[0]
    c.D_kwargs.InputChannels = 3
    c.D_kwargs.MLPWidthRatio = 2
    c.D_kwargs.AttentionWidthRatio = 1
    c.D_kwargs.ChannelsPerAttentionHead = 64
    c.D_kwargs.NumberOfClasses = 10


    
    
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.ema_snapshot_ticks = opts.ema_snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.g_batch_gpu is not None and c.d_batch_gpu is not None:
        if c.batch_size % (c.num_devices * c.g_batch_gpu) != 0 or c.batch_size % (c.num_devices * c.d_batch_gpu) != 0:
            raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')

            
    # Augmentation.
    if opts.aug:
        c.augment_kwargs = dnnlib.EasyDict(**aug_config)
        
    c.ema_kwargs = dnnlib.EasyDict(stds=ema_stds)

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume

    # Performance-related toggles.
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{dataset_name:s}-gpus{c.num_devices:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    
    desc=desc
    outdir=opts.outdir
    tpu_id=opts.tpu_id
    
    prefix = f'{tpu_id}-' if tpu_id else ''
    pattern = rf'^{re.escape(tpu_id)}-(\d+)' if tpu_id else r'^(\d+)'
    prev_run_dirs = os.listdir(outdir) if os.path.isdir(outdir) else []
    prev_run_ids = [int(m.group(1)) for x in prev_run_dirs if os.path.isdir(os.path.join(outdir, x)) and (m := re.match(pattern, x))]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{prefix}{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)


    # Create output directory.
    if rank == 0:
        print('Creating output directory...')
        os.makedirs(c.run_dir)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)


    # Launch.
    if rank == 0:
        #print('Launching processes...')
        print("=" * 60, flush=True)
        print("  R3GAN2 Training Config", flush=True)
        print("=" * 60, flush=True)
        print(f"  Preset:              {opts.preset}", flush=True)
        print(f"  Hosts:               {jax.process_count()}", flush=True)
        print(f"  Local devices:       {jax.local_device_count()}", flush=True)
        print(f"  Total devices:       {jax.device_count()}", flush=True)
        print(f"  Devices:             {[str(d) for d in jax.local_devices()]}", flush=True)
        print(f"  Batch size (total):  {c.batch_size}", flush=True)
        print(f"  D batch/device:      {c.d_batch_gpu}", flush=True)
        print(f"  G batch/device:      {c.g_batch_gpu}", flush=True)
        print(f"  Z dim:               {c.G_kwargs.NoiseDimension}", flush=True)
        print(f"  Total kimg:          {c.total_kimg}", flush=True)
        print(f"  Dataset path:        {c.training_set_kwargs.path}", flush=True)
        print(f"  Dataset size:        {c.training_set_kwargs.max_size} images", flush=True)
        print(f"  Dataset resolution:  {c.training_set_kwargs.resolution}", flush=True)
        print(f"  Dataset labels:      {c.training_set_kwargs.use_labels}", flush=True)
        print(f"  Augmentation:        {'enabled' if opts.aug else 'disabled'}", flush=True)
        print(f"  Metrics:             {c.metrics if c.metrics else 'none'}", flush=True)
        print(f"  Resume pkl:          {c.get('resume_pkl', None)}", flush=True)
        print(f"  LR scheduler:        {c.lr_scheduler}", flush=True)
        print(f"  Gamma scheduler:     {c.gamma_scheduler}", flush=True)
        print(f"  Beta2 scheduler:     {c.beta2_scheduler}", flush=True)
        print("=" * 60, flush=True)
        print(flush=True)
        
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
