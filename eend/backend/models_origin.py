#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Copyright 2025 Human Interface Lab (author: C. Moon)
# Licensed under the MIT license.

from os.path import isfile, join

from backend.losses import (
    pit_loss_multispk,
    vad_loss,
)
from backend.updater import (
    NoamOpt,
    setup_optimizer,
)
from pathlib import Path
from torch.nn import Module, ModuleList
from types import SimpleNamespace
from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import os

"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""

class StateChangeDetector(Module):
    def __init__(self, n_units: int, dropout: float = 0.1, device: torch.device = torch.device("cpu")):
        """
        CNN-based State Change Detector

        Args:
            n_units (int): input emb dim
            dropout (float): drop out percentage
            device (torch.device): device
        """
        super(StateChangeDetector, self).__init__()
        self.device = device
        self.detector_layer_1 = torch.nn.Conv1d(n_units, n_units // 2, kernel_size=7, padding=3)
        self.detector_layer_2 = torch.nn.Linear(n_units // 2, n_units // 4)
        self.final_projection = torch.nn.Linear(n_units // 4, 1)
        self.dropout = dropout
        
        self.to(device)

    def forward(self, xs: torch.Tensor) -> torch.Tensor: #def dummy():
        """ Calculate State Change Probability """
        # print("already padded")
        # print("xs.shape: ", xs.shape)
        xs_transposed = xs.permute(0, 2, 1)  # (B, T, D) → (B, D, T)
        # print("after transpose")
        # print("xs_transposed.shape: ", xs_transposed.shape)
        h = torch.tanh(self.detector_layer_1(xs_transposed))
        # print("after first layer")
        # print("h.shape: ", h.shape)
        h = h.permute(0, 2, 1)  # (B, D/2, T) → (B, T, D/2)
        
        # Using reshape
        h = torch.tanh(self.detector_layer_2(h.reshape(-1, h.shape[-1])))
        # print("after second layer")
        # print("h.shape: ", h.shape)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.final_projection(h)
        # print("after final projection")
        # print("h.shape: ", h.shape)
        h = h.view(xs.shape[0], xs.shape[1], -1)
        # print("after view")
        # print("h.shape: ", h.shape)
        
        return h.squeeze(dim=-1)  # (B, T) 

class EncoderDecoderAttractor(Module):
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        encoder_dropout: float,
        decoder_dropout: float,
        detach_attractor_loss: bool,
    ) -> None:
        super(EncoderDecoderAttractor, self).__init__()
        self.device = device
        self.encoder = torch.nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=encoder_dropout,
            batch_first=True,
            device=self.device)
        self.decoder = torch.nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=decoder_dropout,
            batch_first=True,
            device=self.device)
        self.counter = torch.nn.Linear(n_units, 1, device=self.device)
        self.n_units = n_units
        self.detach_attractor_loss = detach_attractor_loss

    def forward(self, xs: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
        _, (hx, cx) = self.encoder.to(self.device)(xs.to(self.device))
        attractors, (_, _) = self.decoder.to(self.device)(
            zeros.to(self.device),
            (hx.to(self.device), cx.to(self.device))
        )
        return attractors

    def estimate(
        self,
        xs: torch.Tensor,
        max_n_speakers: int = 15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attractors from embedding sequences
         without prior knowledge of the number of speakers
        Args:
          xs: List of (T,D)-shaped embeddings
          max_n_speakers (int)
        Returns:
          attractors: List of (N,D)-shaped attractors
          probs: List of attractor existence probabilities
        """
        zeros = torch.zeros((xs.shape[0], max_n_speakers, self.n_units))
        attractors = self.forward(xs, zeros)
        probs = [torch.sigmoid(
            torch.flatten(self.counter.to(self.device)(att)))
            for att in attractors]
        return attractors, probs

    def __call__(
        self,
        xs: torch.Tensor,
        n_speakers: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attractors and loss from embedding sequences
        with given number of speakers
        Args:
          xs: List of (T,D)-shaped embeddings
          n_speakers: List of number of speakers, or None if the number
                                of speakers is unknown (ex. test phase)
        Returns:
          loss: Attractor existence loss
          attractors: List of (N,D)-shaped attractors
        """

        max_n_speakers = max(n_speakers)
        if self.device == torch.device("cpu"):
            zeros = torch.zeros(
                (xs.shape[0], max_n_speakers + 1, self.n_units))
            labels = torch.from_numpy(np.asarray([
                [1.0] * n_spk + [0.0] * (1 + max_n_speakers - n_spk)
                for n_spk in n_speakers]))
        else:
            zeros = torch.zeros(
                (xs.shape[0], max_n_speakers + 1, self.n_units),
                device=torch.device("cuda"))
            labels = torch.from_numpy(np.asarray([
                [1.0] * n_spk + [0.0] * (1 + max_n_speakers - n_spk)
                for n_spk in n_speakers])).to(torch.device("cuda"))

        attractors = self.forward(xs, zeros)
        if self.detach_attractor_loss:
            attractors = attractors.detach()
        logit = torch.cat([
            torch.reshape(self.counter(att), (-1, max_n_speakers + 1))
            for att, n_spk in zip(attractors, n_speakers)])
        loss = F.binary_cross_entropy_with_logits(logit, labels)

        # The final attractor does not correspond to a speaker so remove it
        attractors = attractors[:, :-1, :]
        return loss, attractors


class MultiHeadSelfAttention(Module):
    """ Multi head self-attention layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.device = device
        self.linearQ = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearK = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearV = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearO = torch.nn.Linear(n_units, n_units, device=self.device)
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        self.att = None  # attention for plot

    def __call__(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        # x: (BT, F)
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) \
            / np.sqrt(self.d_k)
        # scores: (B, h, T, T)
        self.att = F.softmax(scores, dim=3)
        p_att = F.dropout(self.att, self.dropout)
        x = torch.matmul(p_att, v.permute(0, 2, 1, 3))
        x = x.permute(0, 2, 1, 3).reshape(-1, self.h * self.d_k)
        return self.linearO(x)


class PositionwiseFeedForward(Module):
    """ Positionwise feed-forward layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        d_units: int,
        dropout: float
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(n_units, d_units, device=self.device)
        self.linear2 = torch.nn.Linear(d_units, n_units, device=self.device)
        self.dropout = dropout

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.dropout(F.relu(self.linear1(x)), self.dropout))


class TransformerEncoder(Module):
    def __init__(
        self,
        device: torch.device,
        idim: int,
        n_layers: int,
        n_units: int,
        e_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.linear_in = torch.nn.Linear(idim, n_units, device=self.device)
        self.lnorm_in = torch.nn.LayerNorm(n_units, device=self.device)
        self.n_layers = n_layers
        self.dropout = dropout
        for i in range(n_layers):
            setattr(
                self,
                '{}{:d}'.format("lnorm1_", i),
                torch.nn.LayerNorm(n_units, device=self.device)
            )
            setattr(
                self,
                '{}{:d}'.format("self_att_", i),
                MultiHeadSelfAttention(self.device, n_units, h, dropout)
            )
            setattr(
                self,
                '{}{:d}'.format("lnorm2_", i),
                torch.nn.LayerNorm(n_units, device=self.device)
            )
            setattr(
                self,
                '{}{:d}'.format("ff_", i),
                PositionwiseFeedForward(self.device, n_units, e_units, dropout)
            )
        self.lnorm_out = torch.nn.LayerNorm(n_units, device=self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) ... batch, time, (mel)freq
        BT_size = x.shape[0] * x.shape[1]
        # e: (BT, F)
        e = self.linear_in(x.reshape(BT_size, -1))
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm1_", i))(e)
            # self-attention
            s = getattr(self, '{}{:d}'.format("self_att_", i))(e, x.shape[0])
            # residual
            e = e + F.dropout(s, self.dropout)
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm2_", i))(e)
            # positionwise feed-forward
            s = getattr(self, '{}{:d}'.format("ff_", i))(e)
            # residual
            e = e + F.dropout(s, self.dropout)
        # final layer normalization
        # output: (BT, F)
        return self.lnorm_out(e)


class TransformerEDADiarization(Module):

    def __init__(
        self,
        device: torch.device,
        in_size: int,
        n_units: int,
        e_units: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        vad_loss_weight: float,
        attractor_loss_ratio: float,
        attractor_encoder_dropout: float,
        attractor_decoder_dropout: float,
        detach_attractor_loss: bool,
    ) -> None: #def dummy():
        """ Self-attention-based diarization model.
        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
          vad_loss_weight (float) : weight for vad_loss
          attractor_loss_ratio (float)
          attractor_encoder_dropout (float)
          attractor_decoder_dropout (float)
        """
        self.device = device
        super(TransformerEDADiarization, self).__init__()
        self.enc = TransformerEncoder(
            self.device, in_size, n_layers, n_units, e_units, n_heads, dropout
        )
        self.eda = EncoderDecoderAttractor(
            self.device,
            n_units,
            attractor_encoder_dropout,
            attractor_decoder_dropout,
            detach_attractor_loss,
        )
        self.attractor_loss_ratio = attractor_loss_ratio
        self.vad_loss_weight = vad_loss_weight

    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor: #def dummy():
        ilens = [x.shape[0] for x in xs]
        # xs: (B, T, F)
        pad_shape = xs.shape
        # emb: (B*T, E)
        emb = self.enc(xs)
        # emb: [(T, E), ...]
        emb = emb.reshape(pad_shape[0], pad_shape[1], -1)
        return emb

    def estimate_sequential(
        self,
        xs: torch.Tensor,
        args: SimpleNamespace
    ) -> List[torch.Tensor]:
        assert args.estimate_spk_qty_thr != -1 or \
            args.estimate_spk_qty != -1, \
            "Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' \
            arguments have to be defined."
        emb = self.get_embeddings(xs)
        ys_active = []
        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractors, probs = self.eda.estimate(
                torch.stack([e[order] for e, order in zip(emb, orders)]))
        else:
            attractors, probs = self.eda.estimate(emb)
        ys = torch.matmul(emb, attractors.permute(0, 2, 1))
        ys = [torch.sigmoid(y) for y in ys]
        for p, y in zip(probs, ys):
            if args.estimate_spk_qty != -1:
                sorted_p, order = torch.sort(p, descending=True)
                ys_active.append(y[:, order[:args.estimate_spk_qty]])
            elif args.estimate_spk_qty_thr != -1:
                silence = np.where(
                    p.data.to("cpu") < args.estimate_spk_qty_thr)[0]
                n_spk = silence[0] if silence.size else None
                ys_active.append(y[:, :n_spk])
            else:
                NotImplementedError(
                    'estimate_spk_qty or estimate_spk_qty_thr needed.')
        return ys_active

    def forward(
        self,
        xs: torch.Tensor,
        ts: torch.Tensor,
        n_speakers: List[int],
        args: SimpleNamespace
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.get_embeddings(xs)

        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractor_loss, attractors = self.eda(
                torch.stack([e[order] for e, order in zip(emb, orders)]),
                n_speakers)
        else:
            attractor_loss, attractors = self.eda(emb, n_speakers)

        # ys: [(T, C), ...]
        ys = torch.matmul(emb, attractors.permute(0, 2, 1))
        return ys, attractor_loss

    def get_loss(
        self,
        ys: torch.Tensor,
        target: torch.Tensor,
        n_speakers: List[int],
        attractor_loss: torch.Tensor,
        vad_loss_weight: float,
        detach_attractor_loss: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_n_speakers = max(n_speakers)
        ts_padded = pad_labels(target, max_n_speakers)
        ts_padded = torch.stack(ts_padded)
        ys_padded = pad_labels(ys, max_n_speakers)
        ys_padded = torch.stack(ys_padded)

        loss = pit_loss_multispk(
            ys_padded, ts_padded, n_speakers, detach_attractor_loss)
        vad_loss_value = vad_loss(ys, target)

        return loss + vad_loss_value * vad_loss_weight + \
            attractor_loss * self.attractor_loss_ratio, loss


def pad_labels(ts: torch.Tensor, out_size: int) -> torch.Tensor:
    # pad label's speaker-dim to be model's n_speakers
    ts_padded = []
    for _, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding
            ts_padded.append(torch.cat((t, -1 * torch.ones((
                t.shape[0], out_size - t.shape[1]))), dim=1))
        elif t.shape[1] > out_size:
            # truncate
            ts_padded.append(t[:, :out_size].float())
        else:
            ts_padded.append(t.float())
    return ts_padded

######################################################################
class TransformerSCDEDADiarization(Module):
    def __init__(
        self,
        device: torch.device,
        in_size: int,
        n_units: int,
        e_units: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        vad_loss_weight: float,
        attractor_loss_ratio: float,
        detach_attractor_loss: float, 
        state_change_detector_dropout: float = 0.1,  
        seg_PIT_loss_ratio: float = 1.0,
        scd_loss_ratio: float = 1.0,
        attractor_encoder_dropout: float = 0.1, 
        attractor_decoder_dropout: float = 0.1, 
    ):
        """ Transformer-based multi-speaker diarization model (EEND-EDA + SSCD) """
        self.device = device

        super(TransformerSCDEDADiarization, self).__init__()
        self.enc = TransformerEncoder(self.device, in_size, n_layers, n_units, e_units, n_heads, dropout)
        self.eda = EncoderDecoderAttractor(
            self.device, n_units, attractor_encoder_dropout, attractor_decoder_dropout, detach_attractor_loss,
        )
        self.scd = StateChangeDetector(device=self.device, n_units=n_units, dropout=state_change_detector_dropout)
        self.attractor_loss_ratio = attractor_loss_ratio
        self.scd_loss_ratio = scd_loss_ratio
        self.seg_PIT_loss_ratio = seg_PIT_loss_ratio

    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor: #def dummy():
        ilens = [x.shape[0] for x in xs]
        # xs: (B, T, F)
        pad_shape = xs.shape
        # emb: (B*T, E)
        emb = self.enc(xs)
        # emb: [(T, E), ...]
        emb = emb.reshape(pad_shape[0], pad_shape[1], -1)
        return emb, ilens
    
    def estimate_sequential(
        self,
        xs: torch.Tensor,
        args: SimpleNamespace
    ) -> List[torch.Tensor]:  # def dummy():

        assert args.estimate_spk_qty_thr != -1 or \
            args.estimate_spk_qty != -1, \
            "Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' \
            arguments have to be defined."
        emb, ilens = self.get_embeddings(xs)
        ys_active = []
        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractors, probs = self.eda.estimate(
                torch.stack([e[order] for e, order in zip(emb, orders)]))
        else:
            attractors, probs = self.eda.estimate(emb)
            
        new_embs = torch.zeros_like(emb, device=xs.device)
        segment_counts = torch.zeros_like(emb, device=xs.device)
        scd_logits = self.scd(emb)
        scd_probs = torch.sigmoid(scd_logits)  # (B, T)
        
        scd_threshold = 0.05
        state_change = scd_probs > scd_threshold
        state_change[:,0] = True
        
        batch_indices = torch.arange(len(ilens), device=xs.device)
        if isinstance(ilens, list):
            ilens = torch.tensor(ilens, device=xs.device)
        last_indices = ilens-1

        state_change[batch_indices, last_indices] = True  
        
        state_changes = torch.cumsum(state_change.int(), dim=1) - 1
        
        new_segment_embs = torch.zeros_like(emb, device=xs.device)
        segment_counts = torch.zeros_like(state_changes, device=xs.device)
        
        segment_lengths = torch.ones_like(state_changes)
        segment_counts = segment_counts.scatter_add(1, state_changes, segment_lengths)
        segment_counts = segment_counts.gather(1, state_changes)

        new_segment_embs = new_segment_embs.scatter_add(1,state_changes.unsqueeze(-1).expand_as(emb), emb)
        new_segment_embs = new_segment_embs.gather(1, state_changes.unsqueeze(-1).expand_as(emb))
        
        new_segment_embs = new_segment_embs / segment_counts.unsqueeze(-1)
                
        ys = torch.matmul(new_segment_embs, attractors.permute(0, 2, 1))
        ys = [torch.sigmoid(y) for y in ys]
        for p, y in zip(probs, ys):
            if args.estimate_spk_qty != -1:
                sorted_p, order = torch.sort(p, descending=True)
                ys_active.append(y[:, order[:args.estimate_spk_qty]])
            elif args.estimate_spk_qty_thr != -1:
                silence = np.where(
                    p.data.to("cpu") < args.estimate_spk_qty_thr)[0]
                n_spk = silence[0] if silence.size else None
                ys_active.append(y[:, :n_spk])
            else:
                NotImplementedError(
                    'estimate_spk_qty or estimate_spk_qty_thr needed.')
        return ys_active

    def forward(
        self,
        xs: torch.Tensor,
        ts: torch.Tensor,
        n_speakers: List[int],
        args: SimpleNamespace,
    ) -> Tuple[torch.Tensor, torch.Tensor]: #def dummy():
        
        emb, ilens = self.get_embeddings(xs)
        if isinstance(ilens, list):
            ilens = torch.tensor(ilens, device=ts.device)
         
        scd_logits = self.scd(emb)  # (B, T, 1)
        scd_probs = torch.sigmoid(scd_logits)  # (B, T)

        
        scd_labels = self.create_state_change_labels(ts, ilens)  # (B, T)
        scd_loss = F.binary_cross_entropy_with_logits(scd_logits, scd_labels)  # (B, T)
        
        scd_threshold = 0.05
        state_change = scd_probs > scd_threshold
        state_change[:,0] = True
        
        batch_indices = torch.arange(len(ilens), device=xs.device)
        last_indices = ilens-1

        state_change[batch_indices, last_indices] = True  
        
        state_changes = torch.cumsum(state_change.int(), dim=1) - 1
        
        new_segment_embs = torch.zeros_like(emb, device=xs.device)
        segment_counts = torch.zeros_like(state_changes, device=xs.device)
        
        segment_lengths = torch.ones_like(state_changes)
        segment_counts = segment_counts.scatter_add(1, state_changes, segment_lengths)
        segment_counts = segment_counts.gather(1, state_changes)
        
        new_segment_embs = new_segment_embs.scatter_add(1,state_changes.unsqueeze(-1).expand_as(emb), emb)
        new_segment_embs = new_segment_embs.gather(1, state_changes.unsqueeze(-1).expand_as(emb))
        
        new_segment_embs = new_segment_embs / segment_counts.unsqueeze(-1)

        # if torch.equal(new_segment_embs, emb) : 
        #     if torch.all(segment_counts == 1):
        #         print("all is same so it is good")
        #     else:
        #         raise ValueError("all is same but it is not good")
        # else:
        #     if torch.all(segment_counts == 1):
        #         raise ValueError("all is not same but it is not good")
        #     else:
        #         print("all is not same so it is good")

        if torch.equal(new_segment_embs, emb):
            if not torch.all(segment_counts == 1):
                raise ValueError("all is same but it is not good")
        else:
            if torch.all(segment_counts == 1):
                raise ValueError("all is not same but it is not good")
        
        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractor_loss, attractors = self.eda(
                torch.stack([e[order] for e, order in zip(emb, orders)]),
                n_speakers)
        else:
            attractor_loss, attractors = self.eda(emb, n_speakers)

        ys = torch.matmul(emb, attractors.permute(0, 2, 1))
        seg_ys = torch.matmul(new_segment_embs, attractors.permute(0, 2, 1))
        
        return ys, seg_ys, attractor_loss, scd_loss

    def get_loss(
        self,
        ys: torch.Tensor,
        target: torch.Tensor,
        n_speakers: List[int],
        attractor_loss: torch.Tensor,
        vad_loss_weight: float,
        detach_attractor_loss: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  #def dummy():

        max_n_speakers = max(n_speakers)
        ts_padded = pad_labels(target, max_n_speakers)
        ts_padded = torch.stack(ts_padded)
        ys_padded = pad_labels(ys, max_n_speakers)
        ys_padded = torch.stack(ys_padded)

        loss = pit_loss_multispk(
            ys_padded, ts_padded, n_speakers, detach_attractor_loss)
        vad_loss_value = vad_loss(ys, target)

        return loss
    
    def create_length_mask(self, length, max_len, num_output):
        batch_size = len(length)
        mask = torch.zeros(batch_size, max_len, num_output, device = length.device)
        for i in range(batch_size):
            mask[i, : length[i], :] = 1
        # mask = to_device(self, mask)
        return mask
    
    def create_state_change_labels(
        self, ts: torch.Tensor, ilens: torch.Tensor, near_n_frames: int = 1
    ) -> torch.Tensor: #def dummy():
        """SSCD Labels generation ( No Gradient )"""

        batch_size, T, C = ts.shape  

        # 🔹 ** Detect Speaker State change ** (B, T-1)
        diff = torch.any(ts[:, 1:] != ts[:, :-1], dim=2) # detect difference beteween previous frame
        scd_labels = torch.cat([torch.zeros(batch_size, 1, device=ts.device), diff.float()], dim=1)

        # 🔹 ** Setting last label as 1 **
        last_valid_idx = ilens - 1  
        mask = torch.zeros_like(scd_labels, dtype=torch.bool)
        mask = mask.scatter(1, last_valid_idx.unsqueeze(1), 1) 
    
        scd_labels = scd_labels.masked_fill(mask, 1)  #  delete .detach() (not necessary)

        # 🔹 ** Using Max Pooling to make neighbor frames as 1( No Gradient )**
        scd_labels = scd_labels.unsqueeze(1)  # (B, 1, T)
        scd_labels = F.max_pool1d(scd_labels, kernel_size=2 * near_n_frames + 1, stride=1, padding=near_n_frames)
        scd_labels = scd_labels.squeeze(1)  # (B, T)

        # ✅ ** prevent Gradient flow **
        scd_labels = scd_labels.detach()

        return scd_labels

def pad_sequence(
    features: List[torch.Tensor],
    labels: List[torch.Tensor],
    seq_len: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features_padded = []
    labels_padded = []
    assert len(features) == len(labels), (
        f"Features and labels in batch were expected to match but got "
        "{len(features)} features and {len(labels)} labels.")
    for i, _ in enumerate(features):
        assert features[i].shape[0] == labels[i].shape[0], (
            f"Length of features and labels were expected to match but got "
            "{features[i].shape[0]} and {labels[i].shape[0]}")
        length = features[i].shape[0]
        if length < seq_len:
            extend = seq_len - length
            features_padded.append(torch.cat((features[i], -torch.ones((
                extend, features[i].shape[1]))), dim=0))
            labels_padded.append(torch.cat((labels[i], -torch.ones((
                extend, labels[i].shape[1]))), dim=0))
        elif length > seq_len:
            raise (f"Sequence of length {length} was received but only "
                   "{seq_len} was expected.")
        else:
            features_padded.append(features[i])
            labels_padded.append(labels[i])
    return features_padded, labels_padded


def _ckpt_path(output_dir: str, epoch: int) -> str:
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    return os.path.join(output_dir, "models", f"checkpoint_{epoch}.tar")

def save_checkpoint(args, epoch, model, optimizer, loss):
    """Save model + optimizer state.
    - If optimizer is a Noam wrapper, also save inner Adam and Noam states.
    """
    path = _ckpt_path(args.output_path, epoch)

    # model state
    model_state = model.state_dict()

    # optimizer state
    # If it looks like a Noam wrapper, it has an inner `.optimizer`
    if hasattr(optimizer, "optimizer"):
        opt_inner_state = optimizer.optimizer.state_dict()   # inner Adam/SGD
        noam_state = optimizer.state_dict()                  # step, warmup, etc.
        ckpt = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": opt_inner_state,
            "noam_state": noam_state,
            "loss": float(getattr(loss, "item", lambda: loss)()),
        }
    else:
        ckpt = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "loss": float(getattr(loss, "item", lambda: loss)()),
        }

    torch.save(ckpt, path)
    return path  # for logging if needed

# === add this helper ===
def _fix_prefix_for_ddp(state_dict: dict, target_state_dict_keys) -> dict:
    has_module_in_ckpt  = any(k.startswith("module.") for k in state_dict.keys())
    has_module_in_model = any(k.startswith("module.") for k in target_state_dict_keys)
    if has_module_in_ckpt and not has_module_in_model:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not has_module_in_ckpt and has_module_in_model:
        return {f"module.{k}": v for k, v in state_dict.items()}
    else:
        return state_dict

def load_checkpoint(args, path, model, optimizer):
    """Load model + optimizer state.
    - Supports both 'model_state' and 'model_state_dict'
    - Aligns DDP 'module.' prefix automatically
    - Restores inner Adam state and Noam wrapper state if present
    Returns: (epoch, model, optimizer, loss)
    """
    ckpt = torch.load(path, map_location=getattr(args, "device", "cpu"))

    # ---- model ----
    raw_model_state = ckpt.get("model_state", None)
    if raw_model_state is None:
        raw_model_state = ckpt.get("model_state_dict", None)
    if raw_model_state is None:
        raise KeyError(f"Checkpoint {path} has neither 'model_state' nor 'model_state_dict'.")

    fixed_state = _fix_prefix_for_ddp(raw_model_state, model.state_dict().keys())
    model.load_state_dict(fixed_state)

    # ---- optimizer ----
    if hasattr(optimizer, "optimizer"):
        # Noam wrapper case
        opt_state = ckpt.get("optimizer_state", None)
        if opt_state is not None:
            optimizer.optimizer.load_state_dict(opt_state)
        noam_state = ckpt.get("noam_state", None)
        if noam_state is not None:
            optimizer.load_state_dict(noam_state)
    else:
        opt_state = ckpt.get("optimizer_state", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    epoch = int(ckpt.get("epoch", 0))
    loss = ckpt.get("loss", None)
    return epoch, model, optimizer, loss


def load_initmodel(args: SimpleNamespace):
    return load_checkpoint(args, args.initmodel)


def get_model(args: SimpleNamespace) -> Module:
    if args.model_type == 'TransformerEDA':
        model = TransformerEDADiarization(
            device=args.device,
            in_size=args.feature_dim * (1 + 2 * args.context_size),
            n_units=args.hidden_size,
            e_units=args.encoder_units,
            n_heads=args.transformer_encoder_n_heads,
            n_layers=args.transformer_encoder_n_layers,
            dropout=args.transformer_encoder_dropout,
            attractor_loss_ratio=args.attractor_loss_ratio,
            attractor_encoder_dropout=args.attractor_encoder_dropout,
            attractor_decoder_dropout=args.attractor_decoder_dropout,
            detach_attractor_loss=args.detach_attractor_loss,
            vad_loss_weight=args.vad_loss_weight,
        )
    elif args.model_type == 'TransformerSCDEDA':
        model = TransformerSCDEDADiarization(
            device=args.device,
            in_size=args.feature_dim * (1 + 2 * args.context_size),
            n_units=args.hidden_size,
            e_units=args.encoder_units,
            n_heads=args.transformer_encoder_n_heads,
            n_layers=args.transformer_encoder_n_layers,
            dropout=args.transformer_encoder_dropout,
            attractor_loss_ratio=args.attractor_loss_ratio,
            attractor_encoder_dropout=args.attractor_encoder_dropout,
            attractor_decoder_dropout=args.attractor_decoder_dropout,
            detach_attractor_loss=args.detach_attractor_loss,
            vad_loss_weight=args.vad_loss_weight,
        )
    else:
        raise ValueError('Possible model_type is "TransformerEDA" or "TransformerSCDEDA')
    return model


def average_checkpoints(device: torch.device, model: Module, models_path: str, epochs: str) -> Module:
    epoch_list = parse_epochs(epochs)  # e.g., "1,2,3" or "0-4"
    states = []
    model_keys = model.state_dict().keys()

    for e in epoch_list:
        ckpt_path = join(models_path, f"checkpoint_{e}.tar")
        ckpt = torch.load(ckpt_path, map_location=device)

        raw_model_state = ckpt.get("model_state", None)
        if raw_model_state is None:
            raw_model_state = ckpt.get("model_state_dict", None)
        if raw_model_state is None:
            raise KeyError(f"{ckpt_path} has neither 'model_state' nor 'model_state_dict'.")

        fixed_state = _fix_prefix_for_ddp(raw_model_state, model_keys)

        tmp = copy.deepcopy(model)
        tmp.load_state_dict(fixed_state)
        states.append(tmp.state_dict())

    avg_state = average_states(states, device)
    out = copy.deepcopy(model)
    out.load_state_dict(avg_state)
    return out


def average_states(
    states_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    qty = len(states_list)
    avg_state = states_list[0]
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] += states_list[i][key].to(device)

    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state


def parse_epochs(string: str) -> List[int]:
    parts = string.split(',')
    res = []
    for p in parts:
        if '-' in p:
            interval = p.split('-')
            res.extend(range(int(interval[0])+1, int(interval[1])+1))
        else:
            res.append(int(p))
    return res