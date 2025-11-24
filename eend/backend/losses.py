#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2025 Human Interface Lab (author: C. Moon)
# Licensed under the MIT license.

from itertools import permutations
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch.nn.functional import logsigmoid
from scipy.optimize import linear_sum_assignment

# ---- global cache for permutation indices ----
_PERM_CACHE = {}  # k(int) -> LongTensor [P_k, k]

def _get_perm_idx(k: int, device: torch.device) -> torch.Tensor:
    # Return cached permutations for k speakers
    if k not in _PERM_CACHE:
        from itertools import permutations
        perms = list(permutations(range(k)))       # length = k!
        _PERM_CACHE[k] = torch.tensor(perms, dtype=torch.long)
    return _PERM_CACHE[k].to(device, non_blocking=True)


def pit_loss_multispk(
        logits: torch.Tensor,               # [B,T,S]
        target: torch.Tensor,               # [B,T,S], -1 = padded
        n_speakers: np.ndarray,             # [B]
        detach_attractor_loss: bool):
    """
    Faster PIT:
    - Split batch by unique k = n_speakers
    - For each group, run PIT on only k speakers (k! perms)
    - All on GPU, no CPU/NumPy/Scipy
    """
    device = logits.device
    B, T, S = logits.shape

    # Make tensors for masks ops
    nspk = torch.as_tensor(n_speakers, device=device, dtype=torch.long)

    # Optional: mask speakers after k with -1 (kept for behavior parity)
    if detach_attractor_loss:
        spk_idx = torch.arange(S, device=device).view(1, 1, S).expand(B, 1, S)
        valid_spk_mask = spk_idx < nspk.view(B, 1, 1)
        target = target.clone()
        target[:, :, ~valid_spk_mask.squeeze(1)] = -1.0

    # Collect losses from each k-group
    all_min_losses = []

    # Unique k values in this batch
    ks = torch.unique(nspk).tolist()
    for k in ks:
        if k <= 0:
            continue  # skip empty
        # Select samples with this k
        idx = (nspk == k)
        if not torch.any(idx):
            continue

        # Slice to first k speakers only (both logits and target)
        # Size: [B_k, T, k]
        log_k = logits[idx][:, :, :k]
        tar_k = target[idx][:, :, :k]

        # Build permutations for k
        perm_idx = _get_perm_idx(k, device)             # [P,k]
        P = perm_idx.shape[0]

        # Reorder target by perms: [P,B_k,T,k]
        tar_exp = tar_k.unsqueeze(0).expand(P, -1, -1, -1)
        idx_exp = perm_idx.view(P, 1, 1, k).expand(-1, tar_k.size(0), T, -1)
        t_perm = torch.gather(tar_exp, 3, idx_exp)

        # Valid mask for BCE: True where target >= 0
        valid = (t_perm >= 0.0)                          # [P,B_k,T,k]

        # BCE with logits per perm, no reduction
        # Logits expand to [P,B_k,T,k]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            log_k.unsqueeze(0).expand(P, -1, -1, -1),
            torch.clamp(t_perm, 0.0, 1.0),
            reduction='none'
        )

        # Mask out invalid positions
        bce = bce * valid

        # Reduce over time and speakers
        num   = bce.sum((2, 3))                          # [P,B_k]
        denom = valid.sum((2, 3)).clamp_min(1)           # [P,B_k]
        loss_perm = num / denom                          # [P,B_k]

        # Min over permutations
        min_loss, _ = loss_perm.min(dim=0)               # [B_k]
        all_min_losses.append(min_loss)

    # Mean over all samples
    return torch.cat(all_min_losses, dim=0).mean()

def sort_loss_multispk(
    logits: torch.Tensor,               # [B, T, S] raw logits
    target: torch.Tensor,               # [B, T, S], -1 means padded speaker
    n_speakers: np.ndarray,             # [B] number of active speakers per item
    detach_attractor_loss: bool,        # kept for API parity (not used here)
):
    """
    Sort-based diarization loss (no PIT):
    - For each sample, sort speaker rows by first active frame (earliest first).
    - Compare model logits to the sorted targets with BCEWithLogits.
    - Mask out padded speakers (-1) and padded frames.
    - Group by k = number of speakers to avoid useless work.
    """

    device = logits.device
    B, T, S = logits.shape
    nspk = torch.as_tensor(n_speakers, device=device, dtype=torch.long)

    # Collect losses from each k-group
    sample_losses = []

    # Unique k values in this batch
    for k in torch.unique(nspk).tolist():
        if k <= 0:
            continue
        idx = (nspk == k)
        if not torch.any(idx):
            continue

        # Slice to first k speakers (common EEND practice)
        log_k = logits[idx][:, :, :k]   # [B_k, T, k]
        tar_k = target[idx][:, :, :k]   # [B_k, T, k]
        Bk = log_k.size(0)

        # Per-sample sort and BCE
        for b in range(Bk):
            y = tar_k[b]                  # [T, k]
            p = log_k[b]                  # [T, k]

            # Frame-valid mask: true where at least one speaker is not padded
            # (all -1 -> padded frame)
            valid_t = (y >= 0.0).any(dim=1)    # [T]

            # Build an "effective" target to find onsets (ignore padded frames)
            y_eff = y.clone()
            y_eff[~valid_t] = 0.0

            # Find first-onset per speaker (first frame with y>0.5)
            onsets = []
            for s_idx in range(k):
                nz = (y_eff[:, s_idx] > 0.5).nonzero(as_tuple=False)
                if nz.numel() == 0:
                    onsets.append(T + 10_000_000)  # very large if never active
                else:
                    onsets.append(int(nz[0, 0].item()))

            # Sort speakers by earliest onset
            order = torch.tensor(onsets, device=device).argsort()  # [k]
            y_sorted = y[:, order]  # [T, k]

            # Valid mask for BCE: keep only non-padded speaker entries and valid frames
            valid_spk = (y_sorted >= 0.0)                 # [T, k]
            valid = valid_spk & valid_t.unsqueeze(1)      # [T, k]

            if not valid.any():
                # nothing valid to compute
                continue

            # BCEWithLogits over valid positions only
            loss_mat = torch.nn.functional.binary_cross_entropy_with_logits(
                p[valid],                              # logits at valid positions
                torch.clamp(y_sorted[valid], 0.0, 1.0),# targets in {0,1}
                reduction='sum'
            )
            denom = valid.sum().clamp_min(1).float()
            sample_losses.append(loss_mat / denom)

    if len(sample_losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(sample_losses).mean()

def vad_loss(ys: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    # Take from reference ts only the speakers that do not correspond to -1
    # (-1 are padded frames), if the sum of their values is >0 there is speech
    vad_ts = (torch.sum((ts != -1)*ts, 2, keepdim=True) > 0).float()
    # We work on the probability space, not logits. We use silence probabilities
    ys_silence_probs = 1-torch.sigmoid(ys)
    # The probability of silence in the frame is the product of the
    # probability that each speaker is silent
    silence_prob = torch.prod(ys_silence_probs, 2, keepdim=True)
    # Estimate the loss. size=[batch_size, num_frames, 1]
    loss = F.binary_cross_entropy(silence_prob, 1-vad_ts, reduction='none')
    # "torch.max(ts, 2, keepdim=True)[0]" keeps the maximum along speaker dim
    # Invalid frames in the sequence (padding) will be -1, replace those
    # invalid positions by 0 so that those losses do not count
    loss[torch.where(torch.max(ts, 2, keepdim=True)[0] < 0)] = 0
    # normalize by sequence length
    # "torch.sum(loss, axis=1)" gives a value per batch
    # if torch.mean(ts,axis=2)==-1 then all speakers were invalid in the frame,
    # therefore we should not account for it
    # ts is size [batch_size, num_frames, num_spks]
    loss = torch.sum(loss, axis=1) / (torch.mean(ts, axis=2) != -1).sum(axis=1, keepdims=True)
    # normalize in batch for all speakers
    loss = torch.mean(loss)
    return loss

