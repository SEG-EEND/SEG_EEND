#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
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


"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""

class StateChangeDetector(Module):
    def __init__(self, n_units: int, dropout: float = 0.1, device: torch.device = torch.device("cpu")):
        """
        CNN 기반 State Change Detector

        Args:
            n_units (int): 입력 임베딩 차원
            dropout (float): 드롭아웃 비율
            device (torch.device): 실행할 디바이스 (기본값: "cpu")
        """
        super(StateChangeDetector, self).__init__()
        self.device = device
        self.detector_layer_1 = torch.nn.Conv1d(n_units, n_units // 2, kernel_size=7, padding=3)
        self.detector_layer_2 = torch.nn.Linear(n_units // 2, n_units // 4)
        self.final_projection = torch.nn.Linear(n_units // 4, 1)
        self.dropout = dropout
        
        self.to(device)

    def forward(self, xs: torch.Tensor) -> torch.Tensor: #def dummy():
        """ State Change Probability 계산 """
        # print("already padded")
        # print("xs.shape: ", xs.shape)
        xs_transposed = xs.permute(0, 2, 1)  # (B, T, D) → (B, D, T)
        # print("after transpose")
        # print("xs_transposed.shape: ", xs_transposed.shape)
        h = torch.tanh(self.detector_layer_1(xs_transposed))
        # print("after first layer")
        # print("h.shape: ", h.shape)
        h = h.permute(0, 2, 1)  # (B, D/2, T) → (B, T, D/2)
        
        # ✅ `view()` 대신 `.reshape()` 사용하여 메모리 연속성 문제 해결
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
        
        return h.squeeze(dim=-1)  # ✅ (B, T) 형태로 반환

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
        detach_attractor_loss: float,  # ✅ 디폴트 값 없음 → 앞에 배치
        state_change_detector_dropout: float = 0.1,  # ✅ 디폴트 값 없음 → 앞에 배치
        seg_PIT_loss_ratio: float = 1.0,
        scd_loss_ratio: float = 1.0,
        attractor_encoder_dropout: float = 0.1,  # ✅ 디폴트 값 있음 → 뒤에 배치
        attractor_decoder_dropout: float = 0.1,  # ✅ 디폴트 값 있음 → 뒤에 배치
    ):
        """ Transformer 기반 다중화자 다이어리제이션 모델 (EEND-EDA + SSCD) """
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
        """
        **GPU 기반 SSCD(State Change Detector) + EEND 다이어리제이션 추론**
        
        Args:
            xs (torch.Tensor): 입력 오디오 임베딩 (B, T, D) (GPU에서 연산)
            args (SimpleNamespace): 설정 값
        
        Returns:
            List[torch.Tensor]: 다이어리제이션 결과 [(T, C), ...]
        """
        assert args.estimate_spk_qty_thr != -1 or args.estimate_spk_qty != -1, \
            "Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' must be defined."

        device = xs.device  # ✅ GPU 디바이스 유지
        emb, scd_logits, ilens = self.forward(xs)
        ys_active = []

        if isinstance(ilens, list):
            ilens = torch.tensor(ilens, device=ts.device)
            
        # ✅ **SSCD 확률 계산 (GPU)**
        scd_probs = torch.sigmoid(scd_logits)  # 🔥 리스트 대신 Tensor 사용 (연산 최적화)

        # ✅ **화자 임베딩 순서 랜덤 셔플링 여부**
        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractors, probs = self.eda.estimate(
                torch.stack([e[order] for e, order in zip(emb, orders)]))
        else:
            attractors, probs = self.eda.estimate(emb)

        # ✅ **SSCD 확률을 기반으로 상태 변화 감지**
        scd_threshold = getattr(args, "scd_threshold", 0.05)  # 🔥 안전한 속성 접근
        refined_predictions = []
        
        for idx in range(len(emb)):
            try:
                len, e, scd_prob, att = ilens[idx], emb[idx], scd_probs[idx], attractors[idx]

                # 🔥 **상태 변화 검출 (GPU에서 처리)**
                state_changes = torch.where(scd_prob > scd_threshold)[0]
                state_changes = torch.cat([torch.tensor([0], device=device), state_changes, torch.tensor([e.shape[0] - 1], device=device)])
                state_changes = torch.unique(state_changes)

                # 🔥 **구간별 평균 임베딩을 계산하여 화자 예측 향상**
                refined_pred = torch.zeros((e.shape[0], att.shape[0]), dtype=torch.float32, device=device)  # (T, C)
                for start, end in zip(state_changes[:-1], state_changes[1:]):
                    segment_emb = e[start:end].mean(dim=0, keepdim=True)  # 평균 임베딩
                    refined_segment = torch.matmul(segment_emb, att.T)  # 구간별 화자 예측
                    refined_pred[start:end] = torch.clamp(refined_segment, 0, 1)  # (T, C)

                refined_predictions.append(refined_pred)
            except Exception as err:
                print(f"🔥 Error in refining predictions for index {idx}: {err}")
                raise
        
        print("✅ type of refined_predictions: ", type(refined_predictions))
        
        # ✅ **화자 수 추정 및 최종 다이어리제이션 결과 생성**
        ys = refined_predictions
        for p, y in zip(probs, ys):
            if args.estimate_spk_qty != -1:
                # 🚀 화자 수 고정 (`estimate_spk_qty`)
                sorted_p, order = torch.sort(p, descending=True)
                ys_active.append(y[:, order[:args.estimate_spk_qty]])
            elif args.estimate_spk_qty_thr != -1:
                # 🚀 특정 확률 임계값(`estimate_spk_qty_thr`) 이하인 화자 제외
                silence = torch.where(p < args.estimate_spk_qty_thr)[0]
                n_spk = silence[0] if silence.numel() > 0 else None
                ys_active.append(y[:, :n_spk])
            else:
                raise NotImplementedError('estimate_spk_qty or estimate_spk_qty_thr must be set.')

        return ys_active

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:  # def dummy():
        """
        **Transformer Encoder + SSCD Forward Pass**
        
        Args:
            xs (torch.Tensor): 입력 오디오 임베딩 (B, T, D)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[int]]: 
                - emb (B, T, D) : Transformer Encoder 출력 임베딩
                - scd_logits (B, T): 상태 변화 검출 로짓 (State Change Detector)
                - ilens (List[int]): 입력 길이 리스트
        """
        emb, ilens = self.get_embeddings(xs)  # 🔥 Transformer Encoder를 통한 임베딩 추출
        scd_logits = self.scd(emb)  # 🔥 상태 변화 검출 로짓 계산

        return emb, scd_logits, ilens

    def get_loss(
        self,
        xs: torch.Tensor,  # (Batch, T, D)
        ts: torch.Tensor,  # (Batch, T, C)
        n_speakers: List[int],
        args: SimpleNamespace,
        detach_attractor_loss: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  #def dummy():

        """ **손실 계산 함수 (Batch × T × C 입력)** """

        emb, scd_logits, ilens = self.forward(xs)  # (Batch, T, D), (Batch, T, 1)
        device = xs.device

        if isinstance(ilens, list):
            ilens = torch.tensor(ilens, device=ts.device)

        # ✅ **Attractor Loss 계산**
        attractor_loss, attractors = self.eda(emb, n_speakers)

        # ✅ **PIT Loss (프레임 단위)**
        ys = torch.matmul(emb, attractors.permute(0, 2, 1))  # (Batch, T, C)
        max_n_speakers = max(n_speakers)
        
        ts_padded = torch.stack(pad_labels(ts, max_n_speakers))  
        ys_padded = torch.stack(pad_labels(ys, max_n_speakers))  
        
        output_mask = self.create_length_mask(ilens, ts_padded.shape[1], ts_padded.shape[2])        
        ys_padded = ys_padded * output_mask
        
        frame_PIT_loss = pit_loss_multispk(ys_padded, ts_padded, n_speakers, detach_attractor_loss)
            
        # ✅ **SSCD Loss (State Change Detector Loss)**    
        scd_mask = self.create_length_mask(ilens, scd_logits.shape[1], 1)
        scd_mask = scd_mask.squeeze(-1)
     
        scd_labels = self.create_state_change_labels(ts_padded, ilens)
        scd_logits = scd_logits * scd_mask
        
        scd_loss = F.binary_cross_entropy_with_logits(scd_logits, scd_labels)

        # ✅ **Segment PIT Loss (SSCD 기반 Segmentation)**
        scd_probs = torch.sigmoid(scd_logits)
        scd_threshold = 0.05  
        state_change_mask = scd_probs > scd_threshold  
        state_change_mask[:, 0] = True  

        batch_indices = torch.arange(len(ilens), device=device)
        last_indices = (ilens - 1).clamp(min=0)
        state_change_mask[batch_indices, last_indices] = True  

        # 🔹 state_changes 추출
        state_changes = torch.nonzero(state_change_mask)[1]  

        # if is_main_process:
        #     print(f"state_changes before filtering: {state_changes}")

        # ✅ 유효한 인덱스만 유지하도록 필터링
        max_valid_index = emb.shape[1]  
        valid_mask = (state_changes >= 0) & (state_changes < max_valid_index)

        # if not valid_mask.all() and is_main_process:
        #     print(f"🚨 Invalid indices found in state_changes! Filtering out invalid values.")

        state_changes = state_changes[valid_mask]

        # ✅ **scatter_add_()로 segment-wise 평균 계산**
        state_changes = torch.cumsum(state_change_mask.int(), dim=1) - 1
    
        state_changes = state_changes.clamp(min=0, max=emb.shape[1] - 1)  

        # if is_main_process:
        #     print(f"state_changes min: {state_changes.min()}, max: {state_changes.max()}, shape: {state_changes.shape}")

        new_segment_embs = torch.zeros_like(emb, device=device)  
        segment_counts = torch.zeros_like(state_changes, dtype=torch.float32, device=device)

        segment_counts = segment_counts.clone().scatter_add(1, state_changes, torch.ones_like(state_changes, dtype=torch.float32))
        # if is_main_process:
        #     print(f"segment_counts min: {segment_counts.min()}, max: {segment_counts.max()}, shape: {segment_counts.shape}")
        #     print(f"Before scatter_add new_segment_embs[0, :5, :] = {new_segment_embs[0, :5, :]}")
        #     print(f"Before scatter_add: segment_counts[0,30] = {segment_counts[0,30]}")

        new_segment_embs = new_segment_embs.clone().scatter_add(1, state_changes.unsqueeze(-1).expand_as(emb), emb)

        # if is_main_process:
        #     print(f"After scatter_add new_segment_embs[0, :5, :] = {new_segment_embs[0, :5, :]}")

        segment_counts = segment_counts.clamp(min=1).clone()  
        new_segment_embs = new_segment_embs / segment_counts.unsqueeze(-1)
                
        # if is_main_process:
        #     print(f"Segment counts unique values: {segment_counts.unique()}")
        #     for i in range(16):
            #     print("segment_counts[i,:] :", segment_counts[i,:])
            # print(f"new_segment_embs[0, :10, :] = {new_segment_embs[0, :10, :]}")
            # print(f"Final new_segment_embs[0,30,:] = {new_segment_embs[0,30,:]}")

        # 🔹 refined_segment 계산
        refined_segment = torch.matmul(new_segment_embs, attractors.permute(0, 2, 1))
        refined_segment = refined_segment.gather(1, state_changes.unsqueeze(-1).expand(-1, -1, refined_segment.shape[2]))
        ys = ys * output_mask
        
        refined_segment = refined_segment * output_mask

        seg_PIT_loss = pit_loss_multispk(refined_segment, ts_padded, n_speakers, detach_attractor_loss)

        print(f"emb shape: {emb.shape}, expanded state_changes shape: {state_changes.unsqueeze(-1).expand_as(emb).shape}")
        print(f"segment_counts max: {segment_counts.max()}, min: {segment_counts.min()}")
        print(f"segment_counts.shape: {segment_counts.shape}")  
        print("state_changes[0,:]: ", state_changes[0,:])
        print("segment_counts[0,:]: ", segment_counts[0,:])
        print("new_segment_embs[0,:30,:]: ", new_segment_embs[0,:30,:])
        print("emb[0,:30,:]: ", emb[0,:30,:])

        # ✅ **최종 손실 합산**
        total_loss = (
            frame_PIT_loss
            + seg_PIT_loss
            # + attractor_loss
            # + scd_loss
        )

        return refined_segment, total_loss, frame_PIT_loss, attractor_loss
    
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
        """SSCD Labels 생성 (Gradient 없음)"""

        batch_size, T, C = ts.shape  

        # 🔹 **상태 변화 감지** (B, T-1)
        diff = torch.any(ts[:, 1:] != ts[:, :-1], dim=2)  # 변화 감지 (더 정확한 방식)
        scd_labels = torch.cat([torch.zeros(batch_size, 1, device=ts.device), diff.float()], dim=1)

        # 🔹 **ilens 위치에 상태 변화(1) 설정**
        last_valid_idx = ilens - 1  
        mask = torch.zeros_like(scd_labels, dtype=torch.bool)
        mask = mask.scatter(1, last_valid_idx.unsqueeze(1), 1) 
    
        scd_labels = scd_labels.masked_fill(mask, 1)  # ✅ .detach() 제거 (필요 없음)

        # 🔹 **Max Pooling을 사용하여 주변 프레임 확장 (Gradient 없음)**
        scd_labels = scd_labels.unsqueeze(1)  # (B, 1, T)
        scd_labels = F.max_pool1d(scd_labels, kernel_size=2 * near_n_frames + 1, stride=1, padding=near_n_frames)
        scd_labels = scd_labels.squeeze(1)  # (B, T)

        # ✅ **Gradient 방지**
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


def save_checkpoint(
    args,
    epoch: int,
    model: Module,
    optimizer: NoamOpt,
    loss: torch.Tensor
) -> None:
    Path(f"{args.output_path}/models").mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        f"{args.output_path}/models/checkpoint_{epoch}.tar"
    )


def load_checkpoint(args: SimpleNamespace, filename: str):
    model = get_model(args)
    optimizer = setup_optimizer(args, model)

    assert isfile(filename), \
        f"File {filename} does not exist."
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
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


def average_checkpoints(
    device: torch.device,
    model: Module,
    models_path: str,
    epochs: str
) -> Module:
    epochs = parse_epochs(epochs)
    states_dict_list = []
    for e in epochs:
        copy_model = copy.deepcopy(model)
        checkpoint = torch.load(join(
            models_path,
            f"checkpoint_{e}.tar"), map_location=device)
        copy_model.load_state_dict(checkpoint['model_state_dict'])
        states_dict_list.append(copy_model.state_dict())
    avg_state_dict = average_states(states_dict_list, device)
    avg_model = copy.deepcopy(model)
    avg_model.load_state_dict(avg_state_dict)
    return avg_model


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
