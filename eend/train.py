#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini)
# Copyright 2025 Human Interface Lab (author: C. Moon)
# Licensed under the MIT license.


from backend.models import (
    average_checkpoints,
    get_model,
    load_checkpoint,
    pad_labels,
    pad_sequence,
    save_checkpoint,
)
from backend.losses import (
    pit_loss_multispk,
    sort_loss_multispk,
    vad_loss,
)
from backend.updater import setup_optimizer, get_rate
from common_utils.diarization_dataset import KaldiDiarizationDataset
from common_utils.gpu_utils import use_single_gpu
from common_utils.metrics import (
    calculate_metrics,
    new_metrics,
    new_metrics_scd,
    reset_metrics,
    update_metrics,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import os

# === torchrun setup ===
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
# === torchrun setup ===  

import random
import torch
import logging
import yamlargparse

# === DEBUG ADDITIONS ===
import sys
import time
from datetime import timedelta
from tqdm.auto import tqdm
import csv
import faulthandler
faulthandler.enable()

# === FASTER TRAINING ===
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")  # use shared files instead of file descriptors
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# === DDP IMPORTS ===
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def setup_ddp(backend: str) -> int:
    """
        DDP init for torchrun only:
        - torchrun sets env vars: RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
        - we just use them and call init_process_group("env://")
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        # single-process / single-GPU
        return 0

    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "To use DDP, you must run with torchrun."
            "e.g., torchrun --nproc_per_node=2 eend/train.py …"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Do not set init_method; use the env (MASTER_ADDR/PORT) provided by torchrun.
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    return local_rank

def cleanup_ddp():
    if is_dist_avail_and_initialized():
        try:
            dist.barrier(timeout=timedelta(seconds=10))
        except Exception:
            pass
        finally:
            try:
                dist.destroy_process_group()
            except Exception:
                pass

# === DDP IMPORTS END ===

# === CHECKPOINT ===

def safe_list_checkpoints(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_") and f.endswith(".tar")],
        key=os.path.getmtime
    )

def safe_load_checkpoint(args, path, model, optimizer, retries: int = 3, wait: float = 0.5):
    last_err = None
    for _ in range(retries):
        try:
            return load_checkpoint(args, path, model, optimizer)
        except OSError as e:
            last_err = e
            time.sleep(wait)
    raise last_err

# === CHECKPOINT ===

def setup_logging(output_path: str):
    os.makedirs(output_path, exist_ok=True)
    log_fp = os.path.join(output_path, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_fp, mode="a", encoding="utf-8"),
        ],
    )
    logging.info(f"[LOG] Writing logs to: {log_fp}")


def log_system_info(args):
    logging.info("[SYS] Python       : %s", sys.version.replace("\n", " "))
    logging.info("[SYS] Torch        : %s", torch.__version__)
    logging.info("[SYS] CUDA (torch) : %s", torch.version.cuda)
    logging.info("[SYS] CUDA avail   : %s", torch.cuda.is_available())
    logging.info("[SYS] cudnn        : enabled=%s, benchmark=%s, deterministic=%s",
                 torch.backends.cudnn.enabled,
                 torch.backends.cudnn.benchmark,
                 torch.backends.cudnn.deterministic)
    if torch.cuda.is_available():
        logging.info("[GPU] device_count: %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logging.info("[GPU] #%d name=%s, total_mem=%.2f GB, sm=%d",
                         i, props.name, props.total_memory/1024/1024/1024, props.multi_processor_count)
        logging.info("[GPU] requested id: %s", args.gpu)
        

def mem_report(tag: str, device: torch.device):
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        logging.info("[MEM][%s] cuda allocated=%.1f MB, reserved=%.1f MB", tag, alloc, reserved)  

# ===== GPU Prefetcher & CPU prepare =====

def prepare_batch_on_cpu(batch, num_frames):
    """Pad and stack on CPU, return CPU tensors."""
    features_list = batch['xs']
    labels_list   = batch['ts']
    n_speakers = np.asarray([
        max(torch.where(t.sum(0) != 0)[0]) + 1 if t.sum() > 0 else 0
        for t in labels_list
    ])
    max_n_speakers = max(n_speakers) if len(n_speakers) else 0
    features_list, labels_list = pad_sequence(features_list, labels_list, num_frames)
    labels_list = pad_labels(labels_list, max_n_speakers)

    features = torch.stack(features_list)   # CPU tensor (equal sizes)
    labels   = torch.stack(labels_list)     # CPU tensor (equal sizes)
    return features, labels, n_speakers, batch.get('names', None)

class CUDAPrefetcher:
    """Pad/stack on CPU, then prefetch to CUDA stream."""
    def __init__(self, loader, device, num_frames):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.num_frames = num_frames
        self.next_batch = None
        self.preload()

    def preload(self):
        try:
            raw = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        # CPU에서 pad→stack 완료
        feats_cpu, labs_cpu, nspk, names = prepare_batch_on_cpu(raw, self.num_frames)
        # 별도 CUDA stream에서 H2D
        with torch.cuda.stream(self.stream):
            feats = feats_cpu.to(self.device, non_blocking=True)
            labs  = labs_cpu.to(self.device, non_blocking=True)
            self.next_batch = {'xs': feats, 'ts': labs, 'nspk': nspk, 'names': names}

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            for k in ('xs', 'ts'):
                batch[k].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch
    
# =======================================
                    
def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _convert(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Dict[str, Any]:
    return {'xs': [x for x, _, _ in batch],
            'ts': [t for _, t, _ in batch],
            'names': [r for _, _, r in batch]}


def compute_loss_and_metrics(
    model_type: str,
    model: torch.nn.Module,
    labels: torch.Tensor,
    input: torch.Tensor,
    n_speakers: List[int],
    acum_metrics: Dict[str, float],
    vad_loss_weight: float,
    detach_attractor_loss: bool
) -> Tuple[torch.Tensor, Dict[str, float]]:

    # unwrap for custom methods/attrs (e.g., get_loss)
    base_model = model.module if hasattr(model, "module") else model

    if model_type in ("TransformerEDA", "ConformerEDA"):
        # forward -> logits and attractor aux
        y_pred, attractor_loss = model(input, labels, n_speakers, args)

        # get PIT-based losses: total and PIT-only
        total_loss, pit_only = base_model.get_loss(
            y_pred, labels, n_speakers, attractor_loss, vad_loss_weight,
            detach_attractor_loss
        )
        extra_terms = total_loss - pit_only  # e.g., vad/attractor parts

        # decide hybrid weights
        pit_w, sort_w = get_hybrid_weights(args)

        # enable sort only if encoder has positional encoding
        enc = base_model.enc if hasattr(base_model, "enc") else None
        posenc_enabled = (enc is not None) and getattr(enc, "pos_enc", None) is not None

        # compute sort loss if enabled
        if sort_w > 0.0 and posenc_enabled:
            nspk_t = torch.as_tensor(n_speakers, device=labels.device)
            s_loss = sort_loss_multispk(y_pred, labels, nspk_t, detach_attractor_loss=False)
        else:
            s_loss = torch.tensor(0.0, device=labels.device)

        # hybrid core and final loss
        hybrid_core = pit_w * pit_only + sort_w * s_loss
        loss = hybrid_core + extra_terms

        # metrics (thresholded for DER)
        metrics = calculate_metrics(labels.detach(), y_pred.detach(), threshold=0.5)

        # optional arrival-order metrics if posenc exists
        if posenc_enabled and 'calculate_metrics_arrival_order' in globals():
            ao = calculate_metrics_arrival_order(labels.detach(), y_pred.detach(), threshold=0.5)
            metrics.update(ao)

        # accumulate
        acum_metrics = update_metrics(acum_metrics, metrics)
        acum_metrics['loss'] += float(loss.item())
        acum_metrics['loss_standard'] += float(pit_only.item())       # PIT-only (before extra)
        acum_metrics['loss_attractor'] += float(attractor_loss.item()) # keep old tracking
        if sort_w > 0.0 and posenc_enabled:
            acum_metrics['loss_sort'] = acum_metrics.get('loss_sort', 0.0) + float(s_loss.item())

        # logging payload
        core_tag = (
            f"Hybrid(λ={base_model.hybrid_sort_weight:.2f})"
            if base_model.hybrid_loss else
            ("Sort" if base_model.use_posenc else "PIT")
        )
        batch_log = {
            "loss": float(loss.item()),
            "DER": float(metrics.get("DER", float("nan"))),
            "DER_arrival": float(metrics.get("DER_arrival", float("nan"))) if 'DER_arrival' in metrics else float('nan'),
            "core": core_tag
        }
        return loss, acum_metrics, batch_log

    elif model_type in ("TransformerSCDEDA", "ConformerSCDEDA"):
        # forward -> logits, segment logits, and aux losses
        y_pred, seg_y_pred, attractor_loss, scd_loss = model(input, labels, n_speakers, args)

        # main stream PIT: total and PIT-only
        total_main, pit_only_main = base_model.get_loss(
            y_pred, labels, n_speakers, attractor_loss, vad_loss_weight,
            detach_attractor_loss
        )
        extra_main = total_main - pit_only_main

        # segment PIT (unchanged)
        seg_total, seg_pit_only = base_model.get_loss(
            seg_y_pred, labels, n_speakers, attractor_loss, vad_loss_weight,
            detach_attractor_loss
        )

        # decide hybrid weights for main stream only
        pit_w, sort_w = get_hybrid_weights(args)
        enc = base_model.enc if hasattr(base_model, "enc") else None
        posenc_enabled = (enc is not None) and getattr(enc, "pos_enc", None) is not None

        if sort_w > 0.0 and posenc_enabled:
            nspk_t = torch.as_tensor(n_speakers, device=labels.device)
            s_loss = sort_loss_multispk(y_pred, labels, nspk_t, detach_attractor_loss=False)
        else:
            s_loss = torch.tensor(0.0, device=labels.device)

        # hybrid on main stream only
        main_core = pit_w * pit_only_main + sort_w * s_loss
        main_loss = main_core + extra_main

        # final loss composition
        loss = main_loss + seg_total + scd_loss + attractor_loss

        # metrics use segment predictions (as before)
        metrics = calculate_metrics(labels.detach(), seg_y_pred.detach(), threshold=0.5)
        acum_metrics = update_metrics(acum_metrics, metrics)
        acum_metrics['loss'] += float(loss.item())
        acum_metrics['loss_standard'] += float(pit_only_main.item())
        acum_metrics['loss_attractor'] += float(attractor_loss.item())
        acum_metrics['loss_scd'] += float(scd_loss.item())
        acum_metrics['loss_seg_PIT'] += float(seg_total.item())
        if sort_w > 0.0 and posenc_enabled:
            acum_metrics['loss_sort'] = acum_metrics.get('loss_sort', 0.0) + float(s_loss.item())

        batch_log = {
            "loss": float(loss.item()),
            "DER": float(metrics.get("DER", float("nan"))),
            "loss_scd": float(scd_loss.item()),
            "loss_seg_PIT": float(seg_total.item()),
            "core": ("Hybrid(λ={:.2f})".format(sort_w)) if (sort_w > 0.0 and posenc_enabled) else "PIT"
        }
        return loss, acum_metrics, batch_log

    else:
        raise ValueError(f"Unknown model type: {model_type}. Please use TransformerEDA or TransformerSCDEDA.")


def get_hybrid_weights(args):
    """Return (pit_w, sort_w). Clamp to [0,1] and sum to 1."""
    if not getattr(args, 'hybrid_loss', False):
        return 1.0, 0.0  # pure PIT
    lam = float(getattr(args, 'hybrid_sort_weight', 0.3))
    lam = max(0.0, min(1.0, lam))
    return (1.0 - lam), lam

def get_training_dataloaders(
    args: SimpleNamespace 
) -> Tuple[DataLoader, DataLoader]: #def dummy():
    
    # === DEBUG ADDITIONS ===
    t0 = time.perf_counter()
    logging.info("[DATA] Building KaldiDiarizationDataset(train) ...")
    # === /DEBUG ADDITIONS ===
    
    train_set = KaldiDiarizationDataset(#def dummy():
        args.train_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
    )

    dev_set = KaldiDiarizationDataset(
        args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
    )
        
    # === DEBUG ADDITIONS ===
    t1 = time.perf_counter()
    logging.info("[DATA] Train dataset ready in %.2fs. __len__=%d", t1 - t0, len(train_set))
    logging.info("[DATA] Building KaldiDiarizationDataset(dev) ...")
    # === /DEBUG ADDITIONS ===
    
    pin = torch.cuda.is_available()

    # === DDP SAMPLERS ===
    use_ddp = getattr(args, "ddp", False) and get_world_size() > 1

    train_sampler = DistributedSampler(train_set, get_world_size(), get_rank(), shuffle=True, drop_last=True) if use_ddp else None
    dev_sampler   = DistributedSampler(dev_set,   get_world_size(), get_rank(), shuffle=False, drop_last=True) if use_ddp else None
                   
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batchsize,
        collate_fn=_convert,
        num_workers=min(max(args.num_workers, 4), 6),  # RECOMMENDED 4~6
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        worker_init_fn=_init_fn,
        pin_memory=pin,
        persistent_workers=False,   
        prefetch_factor=2,         
    )
    
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.dev_batchsize,
        collate_fn=_convert,
        num_workers=1,
        sampler=dev_sampler,
        shuffle=False,
        worker_init_fn=_init_fn,
        pin_memory=pin,
    )
    
    # === DEBUG ADDITIONS ===
    t2 = time.perf_counter()
    logging.info("[DATA] Dev dataset ready in %.2fs. __len__=%d", t2 - t1, len(dev_set))
    # === /DEBUG ADDITIONS ===

    Y_train, _, _ = train_set.__getitem__(0)
    Y_dev, _, _ = dev_set.__getitem__(0)
    assert Y_train.shape[1] == Y_dev.shape[1], \
        f"Train features dimensionality ({Y_train.shape[1]}) and \
        dev features dimensionality ({Y_dev.shape[1]}) differ."
    assert Y_train.shape[1] == (
        args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of {args.feature_dim} \
        but {Y_train.shape[1]} found."


    # === DEBUG ADDITIONS ===
    t3 = time.perf_counter()
    logging.info("[DATA] DataLoaders ready in %.2fs (total %.2fs).", t3 - t2, t3 - t0)
    # === /DEBUG ADDITIONS ===
    
    return train_loader, dev_loader


def parse_arguments() -> SimpleNamespace: #def dummy():
    parser = yamlargparse.ArgumentParser(description='EEND training')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--dev-batchsize', default=1, type=int,
                        help='number of utterances in one development batch')
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', default=-1, type=int,
                        help='gradient clipping. if < 0, no clipping')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--init-epochs', type=str, default='',
                        help='Initialize model with average of epochs \
                        separated by commas or - for intervals.')
    parser.add_argument('--init-model-path', type=str, default='',
                        help='Initialize the model from the given directory')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max-epochs', type=int,
                        help='Max. number of epochs to train')
    parser.add_argument('--min-length', default=0, type=int,
                        help='Minimum number of frames for the sequences'
                             ' after downsampling.')
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--noam-warmup-steps', default=100000, type=float)
    parser.add_argument('--num-frames', default=500, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int,
                        help='maximum number of speakers allowed')
    parser.add_argument('--num-workers', default=1, type=int,
                        help='number of workers in train DataLoader')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--train-batchsize', default=1, type=int,
                        help='number of utterances in one train batch')
    parser.add_argument('--train-data-dir', type = str,
                        help='kaldi-style data dir used for training.')
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--use-last-samples', default=True, type=bool)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    parser.add_argument('--valid-data-dir',
                        help='kaldi-style data dir used for validation.')

    ## parser for Conformer
    parser.add_argument('--conformer-depthwise-kernel-size', type=int, default=32,
                    help='depthwise conv kernel size for Conformer encoder (paper best: 32)')
    
    ## PARSER FOR ATTRACTOR
    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument(
        '--attractor-loss-ratio', default=1.0, type=float,
        help='weighting parameter')
    attractor_args.add_argument(
        '--attractor-encoder-dropout', type=float)
    attractor_args.add_argument(
        '--attractor-decoder-dropout', type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', type=bool,
        help='If True, avoid backpropagation on attractor loss')
    
    # ADDITIONAL ARGS
    parser.add_argument('--prefetch', action='store_true',
                        help='USE CUDA PREFETCHER')
    parser.add_argument('--ddp', action='store_true',
                    help='Use multi-GPU DistributedDataParallel (torchrun).')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='DDP backend (default: nccl)')
    parser.add_argument('--accum-steps', default=1, type=int,
                        help='number of steps to accumulate gradients before optimizer.step()')
    
    parser.add_argument(
        '--noam-scale-rule',
        default='sqrt',
        type=str,
        choices=['linear', 'sqrt', 'step', 'hybrid'],  # <--- 'hybrid' is added
        help='LR scaling rule for Noam optimizer in DDP.'
    )
    parser.add_argument(
        '--noam-scale-warmup',
        action='store_true',
        help='If set, scales down warmup steps. For stability, not setting this is recommended.'
    )
    
    # for sort loss
    parser.add_argument('--use-posenc', action='store_true', help='Enable positional encoding (SortLoss will be used).')
    parser.add_argument('--hybrid-loss', action='store_true',
                        help='Enable hybrid loss: PIT + Sort (weights sum to 1).')
    parser.add_argument('--hybrid-sort-weight', type=float, default=0.5,
                        help='Weight for Sort loss (lambda in [0,1]). PIT gets (1-lambda).')
    
    # --- Noam scalling options ---
    parser.add_argument('--noam-k', type=int, default=0,
                        help='Override effective k (= world_size × accum_steps). If 0, use automatic value.')
    parser.add_argument('--noam-alpha', type=float, default=None,
                        help='Force lr_scale = kalpha. If None, use the rule (linear/sqrt/hybrid/step).')
    parser.add_argument('--noam-lr-scale', type=float, default=None,
                        help='Set lr_scale manually (e.g., 2.0). If set, rule/alpha are ignored.')
    parser.add_argument('--noam-step-scale', type=int, default=0,
                        help='Override step_scale (schedule multiplier). If 0, follow the rule.')
    parser.add_argument('--noam-warmup-scale', type=float, default=1.0,
                        help='warmup_steps reduced by 1/noam_warmup_scale (e.g., 2.0 → warmup/2)')
    args = parser.parse_args()
    return args


if __name__ == '__main__': #def dummy():
    args = parse_arguments()

    # === DEBUG ADDITIONS ===
    setup_logging(args.output_path if args.output_path else "./outputs")
    logging.info("Using seed value: %s", args.seed)
    # === /DEBUG ADDITIONS ===
    
    # For reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # DDP (DON'T USE SAFE GPU)
    
    if args.ddp:
        local_rank = setup_ddp(args.dist_backend)
        args.device = torch.device(f"cuda:{local_rank}")
    else:
        if args.gpu >= 0 and torch.cuda.is_available():
            # IF NOT DDP, USE SAFE_GPU
            gpu_owner = use_single_gpu(args.gpu)
            logging.info('[GPU] device reserved by safe_gpu: %s', gpu_owner)
            args.device = torch.device("cuda", 0)
        else:
            args.device = torch.device("cpu")
            logging.info('[CPU] Using CPU.')


    if is_main_process():
        log_system_info(args)

    # TensorBoard IS ONLY AT RANK 0
    
    writer = SummaryWriter(f"{args.output_path}/tensorboard") if is_main_process() else None
    
    # === /DEBUG ADDITIONS ===

    train_loader, dev_loader = get_training_dataloaders(args)
    mem_report("after_dataloaders", args.device)
    
    
    if args.init_model_path == '':
        model = get_model(args)
        logging.info("[MODEL] Initialized new model/optimizer.")
    else:
        model = get_model(args)
        model = average_checkpoints(
            args.device, model, args.init_model_path, args.init_epochs)
        logging.info("[MODEL] Loaded averaged checkpoints from %s", args.init_model_path)

    # move to device BEFORE wrapping
    model.to(args.device)

    # tell which loss will be used (PIT or Sort) based on encoder.pos_enc
    enc = model.module.enc if hasattr(model, "module") else model.enc
    use_sort = hasattr(enc, "pos_enc") and (enc.pos_enc is not None)
    logging.info("[MODEL] PosEnc=%s -> %s", str(use_sort), "SortLoss" if use_sort else "PIT")
    
    # --- DDP wrap (MUST AFTER LOADING CHECKPOINT) ---
    if args.ddp:
        model = DDP(
            model,
            device_ids=[args.device.index],
            output_device=args.device.index,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # === SET OPTIMIZER AFTER DDP ===
    optimizer = setup_optimizer(args, model)

    # scale LR only for non-Noam optimizers in DDP
    if args.ddp and get_world_size() > 1 and args.optimizer in ("adam", "sgd"):
        args.lr = args.lr * get_world_size()  # simple linear scaling
    
    if args.model_type in ("TransformerEDA","ConformerEDA"):
        acum_train_metrics = new_metrics()
        acum_dev_metrics = new_metrics()
    elif args.model_type in ("TransformerSCDEDA", "ConformerSCDEDA"):
        acum_train_metrics = new_metrics_scd()
        acum_dev_metrics = new_metrics_scd()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # === CHECK THE CHECK POINT AND LOAD ===
    ckpt_dir = os.path.abspath(os.path.join(args.output_path, 'models'))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = safe_list_checkpoints(ckpt_dir)

    # Unwrap for saving/loading
    model_to_save = model.module if hasattr(model, "module") else model

    if not ckpts:
        # INITIALIZE
        init_epoch = 0
        if is_main_process():
            save_checkpoint(args, init_epoch, model_to_save, optimizer, 0.0)
            logging.info("[CKPT] Saved fresh checkpoint_0.tar")
        # WAIT FOR RANKS
        if is_dist_avail_and_initialized():
            dist.barrier()
    else:
        # RESUME
        if is_dist_avail_and_initialized():
            dist.barrier()
        latest = ckpts[-1]
        epoch, model, optimizer, _ = safe_load_checkpoint(args, latest, model, optimizer)
        init_epoch = epoch
        logging.info("[CKPT] Resuming from %s (epoch=%d)", latest, epoch)
    
    train_batches_qty = len(train_loader)
    dev_batches_qty = len(dev_loader)
    if is_main_process():
        logging.info(f"#batches quantity for train: {train_batches_qty}")
        logging.info(f"#batches quantity for dev: {dev_batches_qty}")
        
        base_model = model.module if hasattr(model, "module") else model
        logging.info(
            f"[MODEL] use_posenc={base_model.use_posenc}, "
            f"hybrid={base_model.hybrid_loss}, "
            f"lambda={getattr(base_model, 'hybrid_sort_weight', None)}"
        )
    
    # === TRAIN LOOP ===
    try:
        for epoch in range(init_epoch, args.max_epochs):
            model.train()
            
            # ---- Gradient accumulation setup (declare once outside the loop) ----
            accum_steps = max(1, int(getattr(args, "accum_steps", 1)))
                        
            # ---- per-epoch accumulators ----
            since_reset_batches = 0
            train_epoch_batches = 0
            train_q_loss_sum = 0.0
            train_q_der_sum  = 0.0
            train_epoch_loss_sum = 0.0
            train_epoch_der_sum  = 0.0


            # === SET QURTER POINTS ===
            quarter_points = {
                int(train_batches_qty * 0.25),
                int(train_batches_qty * 0.50),
                int(train_batches_qty * 0.75),
                train_batches_qty
            }
        
            # === In DDP: Setting the Sampler Seed for Each Epoch ===
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            
            if is_main_process():                
                logging.info("=== [EPOCH %d/%d] START ===", epoch+1, args.max_epochs)
            epoch_start = time.perf_counter()
            batch_times = []
            
            if is_main_process():
                pbar = tqdm(total=train_batches_qty, desc=f"Epoch {epoch+1}/{args.max_epochs}", dynamic_ncols=True)
            else:
                pbar = None
                
            log_every = max(int(args.log_report_batches_num), 1)

            optimizer.zero_grad(set_to_none=True)
            
            # ===== Prefetcher on CUDA =====
            use_prefetch = (args.device.type == "cuda")
            
            if use_prefetch:
                prefetcher = CUDAPrefetcher(train_loader, args.device, args.num_frames)

                batch = prefetcher.next()
                i = 0
                                        
                while batch is not None:
                    t0 = time.perf_counter()

                    # Already on GPU and padded/stacked on CPU
                    features = batch['xs']      # [B, T, D] on CUDA
                    labels   = batch['ts']      # [B, T, S] on CUDA
                    n_speakers = batch['nspk']  # numpy array/list
                    t1 = time.perf_counter()
                    
                    loss, acum_train_metrics, batch_log = compute_loss_and_metrics(
                        args.model_type, model, labels, features, n_speakers, acum_train_metrics,
                        args.vad_loss_weight, args.detach_attractor_loss)
                    t2 = time.perf_counter()
                   

                    # ---- loss backward ----
                    scaled_loss = loss / accum_steps

                    done = i + 1
                    is_last_batch = (done == train_batches_qty)
                    use_ddp_nosync = hasattr(model, "no_sync") and ((done % accum_steps) != 0) and (not is_last_batch)

                    if use_ddp_nosync:
                        with model.no_sync():
                            scaled_loss.backward()
                    else:
                        scaled_loss.backward()

                    # ---- step/clip timing: every accum_steps or at the last batch ----
                    do_step = (((i + 1) % accum_steps) == 0) or (done == train_batches_qty)
                    t3 = time.perf_counter()
                    opt_ms_local = None
                    
                    if do_step:
                        # Only clip once
                        if getattr(args, "gradclip", 0) and args.gradclip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
                        
                        if torch.cuda.is_available():
                            evs = torch.cuda.Event(enable_timing=True)
                            eve = torch.cuda.Event(enable_timing=True)
                            evs.record()
                            optimizer.step()
                            eve.record()
                            torch.cuda.synchronize()
                            opt_ms_local = evs.elapsed_time(eve)
                        else:
                            optimizer.step()

                        # --- Log LR only at actual steps ---
                        if is_main_process() and writer is not None:
                            global_step = epoch * train_batches_qty + done
                            writer.add_scalar("lrate", get_rate(optimizer), global_step)  
                                              
                        optimizer.zero_grad(set_to_none=True)  
                           
                    t4 = time.perf_counter()

                    dt_collate = t1 - t0
                    dt_fwd     = t2 - t1
                    dt_bwd     = t3 - t2
                    dt_step    = t4 - t3
                    bt         = t4 - t0

                    batch_times.append(bt)
                    avg_bt = (sum(batch_times[-log_every:]) /
                              max(len(batch_times[-log_every:]), 1))
                    done = i + 1
                    remain = train_batches_qty - done
                    eta_sec = int(remain * avg_bt)

                    def pct(x): return f"{(x/bt*100):.0f}%"
                    if pbar is not None:
                        pbar.set_postfix(
                            loss=f"{batch_log.get('loss', float('nan')):.4f}",
                            der=f"{batch_log.get('DER',  float('nan')):.4f}",
                            core=batch_log.get('core', 'PIT'),
                            bt=f"{bt:.2f}s",
                            eta=f"{eta_sec}s",
                            step=f"{opt_ms_local:.1f}ms" if opt_ms_local is not None else "n/a",
                            c=pct(dt_collate), f=pct(dt_fwd), b=pct(dt_bwd), s=pct(dt_step)
                        )
                        pbar.update(1)

                    # --- update counters ---
                    since_reset_batches += 1
                    train_epoch_batches += 1

                    train_q_loss_sum     += float(batch_log.get("loss", 0.0))
                    train_q_der_sum      += float(batch_log.get("DER",  0.0))
                    train_epoch_loss_sum += float(batch_log.get("loss", 0.0))
                    train_epoch_der_sum  += float(batch_log.get("DER",  0.0))

                    # --- quarter logging ---
                    if is_main_process() and done in quarter_points:
                        global_step = epoch * train_batches_qty + done
                        q_avg_loss = train_q_loss_sum / max(1, since_reset_batches)
                        q_avg_der  = train_q_der_sum  / max(1, since_reset_batches)

                        writer.add_scalar("train_loss_qavg", q_avg_loss, global_step)
                        writer.add_scalar("train_DER_qavg",  q_avg_der,  global_step)

                        logging.info(
                            "[EP%02d] step %d/%d (%.0f%%) | loss=%.4f | DER=%.4f",
                            epoch+1, done, train_batches_qty, (done/train_batches_qty)*100,
                            q_avg_loss, q_avg_der
                        )

                        # reset only quarter accumulators
                        train_q_loss_sum = 0.0
                        train_q_der_sum  = 0.0
                        since_reset_batches = 0

                    batch = prefetcher.next()
                    i += 1
            else:
                # ===== CPU only fallback  =====
                for i, batch in enumerate(train_loader):
                    t0 = time.perf_counter()
                    features = batch['xs']
                    labels = batch['ts']
                    n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                             if t.sum() > 0 else 0 for t in labels])
                    max_n_speakers = max(n_speakers)
                    features, labels = pad_sequence(features, labels, args.num_frames)
                    labels = pad_labels(labels, max_n_speakers)
                    t1 = time.perf_counter()

                    features = torch.stack(features).to(args.device, non_blocking=True)
                    labels = torch.stack(labels).to(args.device, non_blocking=True)
                    if args.device.type == "cuda":
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()

                    loss, acum_train_metrics, batch_log = compute_loss_and_metrics(
                        args.model_type, model, labels, features, n_speakers, acum_train_metrics,
                        args.vad_loss_weight, args.detach_attractor_loss)
                    t3 = time.perf_counter()

                    # loss backward (accumulated scale)
                    scaled_loss = loss / accum_steps
                    done = i + 1
                    is_last_batch = (done == train_batches_qty)

                    scaled_loss.backward()

                    do_step = (done % accum_steps == 0) or is_last_batch
                    t4 = time.perf_counter()
                    opt_ms_local = None

                    if do_step:
                        if getattr(args, "gradclip", 0) and args.gradclip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)

                        if torch.cuda.is_available():
                            evs = torch.cuda.Event(enable_timing=True)
                            eve = torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize()
                            evs.record(); optimizer.step(); eve.record()
                            torch.cuda.synchronize()
                            opt_ms_local = evs.elapsed_time(eve)
                        else:
                            optimizer.step()

                        if is_main_process() and writer is not None:
                            global_step = epoch * train_batches_qty + done
                            writer.add_scalar("lrate", get_rate(optimizer), global_step)

                        optimizer.zero_grad(set_to_none=True)
                        
                    t5 = time.perf_counter()

                    dt_collate = t1 - t0
                    dt_h2d    = t2 - t1
                    dt_fwd    = t3 - t2
                    dt_bwd    = t4 - t3
                    dt_step   = t5 - t4
                    bt        = t5 - t0

                    batch_times.append(bt)
                    avg_bt = (sum(batch_times[-log_every:]) /
                              max(len(batch_times[-log_every:]), 1))
                    done = i + 1
                    remain = train_batches_qty - done
                    eta_sec = int(remain * avg_bt)

                    def pct(x): return f"{(x/bt*100):.0f}%"
                    if pbar is not None:
                        pbar.set_postfix(
                            loss=f"{batch_log.get('loss', float('nan')):.4f}",
                            der=f"{batch_log.get('DER', float('nan')):.4f}",
                            core=batch_log.get('core', 'PIT'),
                            bt=f"{bt:.2f}s",
                            eta=f"{eta_sec}s",
                            step=f"{opt_ms_local:.1f}ms" if opt_ms_local is not None else "n/a",
                            c=pct(dt_collate), f=pct(dt_fwd), b=pct(dt_bwd), s=pct(dt_step)
                        )
                        pbar.update(1)


                    # === update counters for averages ===
                    since_reset_batches += 1
                    train_epoch_batches += 1
                    
                    train_q_loss_sum     += float(batch_log.get("loss", 0.0))
                    train_q_der_sum      += float(batch_log.get("DER",  0.0))
                    
                    train_epoch_loss_sum += float(batch_log.get("loss", 0.0))
                    train_epoch_der_sum  += float(batch_log.get("DER",  0.0))

                    # === quarter logging ===
                    if is_main_process() and done in quarter_points:
                        global_step = epoch * train_batches_qty + done
                        q_avg_loss = train_q_loss_sum / max(1, since_reset_batches)
                        q_avg_der  = train_q_der_sum  / max(1, since_reset_batches)

                        writer.add_scalar("train_loss_qavg", q_avg_loss, global_step)
                        writer.add_scalar("train_DER_qavg",  q_avg_der,  global_step)
                        writer.add_scalar("lrate", get_rate(optimizer), global_step)

                        logging.info(
                            "[EP%02d] step %d/%d (%.0f%%) | loss=%.4f | DER=%.4f",
                            epoch+1, done, train_batches_qty, (done/train_batches_qty)*100,
                            q_avg_loss, q_avg_der
                        )

                        # reset only quarter accumulators
                        train_q_loss_sum = 0.0
                        train_q_der_sum  = 0.0
                        since_reset_batches = 0
                    
            if pbar is not None:
                pbar.close()

            if is_main_process():
                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint(args, epoch+1, model_to_save, optimizer, loss)
                
            if is_main_process():
                epoch_time = time.perf_counter() - epoch_start
                loss_train_epoch = train_epoch_loss_sum / max(1, train_epoch_batches)
                der_train_epoch  = train_epoch_der_sum  / max(1, train_epoch_batches)

                writer.add_scalar("train_loss_epoch", loss_train_epoch, epoch+1)
                writer.add_scalar("train_DER_epoch",  der_train_epoch,  epoch+1)
                
                logging.info("=== [EPOCH %d] END | time=%.2fs | loss=%.4f | DER=%.4f ===",
                            epoch+1, epoch_time, loss_train_epoch, der_train_epoch)
                            
            # ===== DEV =====
            with torch.no_grad():
                model.eval()
                
                # In DDP, a dev sampler may also exist (no shuffling).
                if isinstance(dev_loader.sampler, DistributedSampler):
                    dev_loader.sampler.set_epoch(epoch)
                    
                dev_start = time.perf_counter()
                
                # track how many dev batches we've seen
                dev_seen = 0

                # INITIALIZE A DEV LOSS AND DEV DER
                dev_loss_sum = 0.0
                dev_der_sum  = 0.0
    
                for i, batch in enumerate(dev_loader):
                    features = batch['xs']
                    labels = batch['ts']
                    n_speakers = np.asarray([
                        max(torch.where(t.sum(0) != 0)[0]) + 1 if t.sum() > 0 else 0
                        for t in labels
                    ])
                    max_n_speakers = max(n_speakers) if len(n_speakers) else 0
                    features, labels = pad_sequence(features, labels, args.num_frames)
                    labels = pad_labels(labels, max_n_speakers)

                    features = torch.stack(features).to(args.device, non_blocking=True)
                    labels   = torch.stack(labels).to(args.device, non_blocking=True)

                    # capture batch_log to accumulate proper averages
                    _, acum_dev_metrics, batch_log = compute_loss_and_metrics(
                        args.model_type, model, labels, features, n_speakers,
                        acum_dev_metrics, args.vad_loss_weight, args.detach_attractor_loss
                    )
                    dev_seen += 1
                    dev_loss_sum += float(batch_log.get("loss", 0.0))
                    dev_der_sum  += float(batch_log.get("DER",  0.0))

                # write dev scalars (guard empty dev set)
                if dev_seen == 0:
                    logging.warning("[DEV] no batches in dev_loader")
                else:
                    dev_avg_loss = dev_loss_sum / dev_seen
                    dev_avg_der  = dev_der_sum  / dev_seen
                    if is_main_process():
                        writer.add_scalar("dev_loss_epoch", dev_avg_loss, epoch+1)
                        writer.add_scalar("dev_DER_epoch",  dev_avg_der,  epoch+1)

                # LOGGING FINAL RESULT AND RESET
                dev_time = time.perf_counter() - dev_start
                if is_main_process():
                    logging.info("[DEV] done in %.2fs | loss=%.4f | DER=%.4f",
                                dev_time, dev_avg_loss if dev_seen else float("nan"),
                                dev_avg_der  if dev_seen else float("nan"))
                
                acum_dev_metrics = reset_metrics(acum_dev_metrics)

    except KeyboardInterrupt:
        if is_main_process():
            logging.warning("[INTERRUPT] Training interrupted by user.")
    finally:
        mem_report("final", args.device)
        if writer is not None and is_main_process():
            writer.close()
        if args.ddp:
            cleanup_ddp()
        if is_main_process():
            logging.info("[DONE] Training script finished.")
