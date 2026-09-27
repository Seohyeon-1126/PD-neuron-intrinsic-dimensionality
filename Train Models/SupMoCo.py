"""Encoder and supervised MoCo loss used for SupMoCo experiments.
Random seeds: 42, 123, 2024.
"""

from typing import List, Tuple

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.checkpoint import checkpoint_sequential

from torch.utils.data import DataLoader, Dataset, Sampler, Subset

SEED = 2024

OUT_DIM = 512

BLOCKS = (2, 2, 2, 3)

DILATIONS = (1, 1, 1, 1)

REFINE_BLOCKS = 1

CKPT_SEGMENTS = 0

EMBED_DIM = 512

PROJ_LAYERS = 2

PROJ_HIDDEN = 2048

PROJ_BN = False

PROJ_DROPOUT = 0.0

USE_L2_NORM_POOL = True

TEMPERATURE = 0.07

MOCO_M = 0.995

QUEUE_SIZE = 65536

def parse_int_list(s: str, n: int) -> Tuple[int, ...]:

    parts = [p.strip() for p in s.split(",") if p.strip()]

    vals = [int(p) for p in parts]

    if len(vals) != n:

        raise ValueError(f"Expected {n} ints, got {len(vals)} from {s!r}")

    return tuple(vals)

def renorm_unit_per_out_channel_(model: nn.Module, eps: float = 1e-12):

    for m in model.modules():

        if isinstance(m, nn.Conv2d):

            w = m.weight.data

            n = w.flatten(1).norm(dim=1, keepdim=True).clamp_min(eps)

            w.div_(n.view(-1, 1, 1, 1))

        elif isinstance(m, nn.Linear):

            w = m.weight.data

            n = w.norm(dim=1, keepdim=True).clamp_min(eps)

            w.div_(n)

def conv2d(in_ch, out_ch, k=3, stride=1, padding=1, dilation=1, bias=True):

    return nn.Conv2d(

        in_ch,

        out_ch,

        kernel_size=k,

        stride=stride,

        padding=padding,

        dilation=dilation,

        bias=bias,

    )

class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dilation=1):

        super().__init__()

        self.c1 = conv2d(

            in_ch, out_ch, 3, 1,

            padding=dilation, dilation=dilation, bias=True

        )

        self.c2 = conv2d(

            out_ch, out_ch, 3, 1,

            padding=dilation, dilation=dilation, bias=True

        )

        self.proj = None

        if in_ch != out_ch:

            self.proj = conv2d(in_ch, out_ch, 1, 1, padding=0, bias=False)


    def forward(self, x):

        identity = x

        x = F.relu(x, inplace=True)

        x = self.c1(x)

        x = F.relu(x, inplace=True)

        x = self.c2(x)

        if self.proj is not None:

            identity = self.proj(identity)

        return x + identity

class Stage(nn.Module):

    def __init__(

        self, in_ch, out_ch, n_blocks, dilation,

        use_ckpt: bool, ckpt_segments: int

    ):

        super().__init__()

        self.use_ckpt = bool(use_ckpt)

        self.ckpt_segments = int(ckpt_segments)


        blocks = [ResBlock(in_ch, out_ch, dilation=dilation)]

        for _ in range(n_blocks - 1):

            blocks.append(ResBlock(out_ch, out_ch, dilation=dilation))

        self.blocks = nn.Sequential(*blocks)


    def forward(self, x):

        if (

            self.use_ckpt

            and self.training

            and self.ckpt_segments > 1

            and len(self.blocks) > 1

        ):

            seg = min(self.ckpt_segments, len(self.blocks))

            return checkpoint_sequential(

                self.blocks, seg, x, use_reentrant=False

            )

        return self.blocks(x)

class Encoder(nn.Module):

    def __init__(

        self,

        blocks=BLOCKS,

        dilations=DILATIONS,

        refine_blocks=REFINE_BLOCKS,

        ckpt_segments=CKPT_SEGMENTS,

    ):

        super().__init__()

        b2, b3, b4, b5 = blocks

        d2, d3, d4, d5 = dilations


        self.stem = nn.Sequential(

            conv2d(3, 64, k=3, stride=2, padding=1, bias=True)

        )

        self.stage2 = Stage(64, 128, b2, d2, False, 1)

        self.stage3 = Stage(128, 256, b3, d3, False, 1)

        self.stage4 = Stage(

            256, 512, b4, d4, True, ckpt_segments

        )

        self.stage5 = Stage(

            512, OUT_DIM, b5, d5, True, ckpt_segments

        )

        self.refine = Stage(

            OUT_DIM, OUT_DIM, int(refine_blocks), 1,

            True, ckpt_segments

        )


        self.trunk = nn.Sequential(

            self.stem,

            self.stage2,

            self.stage3,

            self.stage4,

            self.stage5,

            self.refine,

        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):

        x = self.trunk(x)

        return self.gap(x).flatten(1)


    @torch.no_grad()

    def forward_feature_maps(self, x, which: str):

        x = self.stem(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)


        if which == "stage5_mid":

            for block in list(self.stage5.blocks)[:-1]:

                x = block(x)

            return x


        x = self.stage5(x)

        if which == "stage5_out":

            return x


        x = self.refine(x)

        if which == "refine_out":

            return x


        raise ValueError(f"Unknown which={which}")

def build_projector(

    in_dim: int,

    embed_dim: int,

    proj_layers: int,

    proj_hidden: int,

    use_bn: bool,

    dropout: float,

):

    def lin(a, b):

        return nn.Linear(a, b, bias=False)


    def bn(d):

        return nn.BatchNorm1d(d) if use_bn else nn.Identity()


    def do():

        return nn.Dropout(dropout) if dropout > 0 else nn.Identity()


    if proj_layers <= 1:

        return nn.Sequential(lin(in_dim, embed_dim))


    if proj_layers == 2:

        return nn.Sequential(

            lin(in_dim, proj_hidden),

            bn(proj_hidden),

            nn.ReLU(inplace=False),

            do(),

            lin(proj_hidden, embed_dim),

        )


    if proj_layers == 3:

        return nn.Sequential(

            lin(in_dim, proj_hidden),

            bn(proj_hidden),

            nn.ReLU(inplace=False),

            do(),

            lin(proj_hidden, proj_hidden),

            bn(proj_hidden),

            nn.ReLU(inplace=False),

            do(),

            lin(proj_hidden, embed_dim),

        )


    raise ValueError(f"Unsupported proj_layers={proj_layers}")

class SupMoCoModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder()

        self.use_l2_norm_pool = USE_L2_NORM_POOL

        self.projector = build_projector(

            OUT_DIM,

            EMBED_DIM,

            PROJ_LAYERS,

            PROJ_HIDDEN,

            PROJ_BN,

            PROJ_DROPOUT,

        )


    def forward(self, x):

        pooled = self.encoder(x)

        if self.use_l2_norm_pool:

            pooled = F.normalize(pooled, dim=1)

        z = self.projector(pooled)

        return F.normalize(z, dim=1)

def momentum_update_(model_q: nn.Module, model_k: nn.Module, m: float):

    for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):

        p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)

class SupervisedMoCoQueue:

    def __init__(self, dim, capacity, device, dtype=torch.float16):

        self.dim = int(dim)

        self.capacity = int(capacity)

        self.device = device

        self.dtype = dtype

        self.reset()


    def reset(self):

        self.ptr = 0

        self.full = False

        self.feats = torch.zeros(

            self.capacity, self.dim,

            device=self.device, dtype=self.dtype

        )

        self.labels = torch.zeros(

            self.capacity, device=self.device, dtype=torch.long

        )


    @torch.no_grad()

    def enqueue(self, feats, labels):

        feats = feats.detach()

        labels = labels.detach()

        b = int(feats.size(0))

        if b <= 0:

            return


        if b > self.capacity:

            feats = feats[-self.capacity:]

            labels = labels[-self.capacity:]

            b = self.capacity


        end = self.ptr + b

        if end <= self.capacity:

            self.feats[self.ptr:end].copy_(feats.to(self.dtype))

            self.labels[self.ptr:end].copy_(labels)

        else:

            first = self.capacity - self.ptr

            second = end - self.capacity

            self.feats[self.ptr:].copy_(feats[:first].to(self.dtype))

            self.labels[self.ptr:].copy_(labels[:first])

            self.feats[:second].copy_(feats[first:].to(self.dtype))

            self.labels[:second].copy_(labels[first:])


        self.ptr = end % self.capacity

        if end >= self.capacity:

            self.full = True


    def get(self):

        if self.full:

            return self.feats, self.labels

        return self.feats[:self.ptr], self.labels[:self.ptr]

def supervised_contrastive_q_vs_k(

    q, y_q, k, y_k, temperature

):

    q = q.float()

    k = k.float()


    logits = (q @ k.t()) / float(temperature)

    logits = logits - logits.max(dim=1, keepdim=True).values


    with torch.no_grad():

        pos = y_q.view(-1, 1) == y_k.view(1, -1)

        pos_cnt = pos.sum(dim=1).clamp_min(1)


    exp_logits = torch.exp(logits)

    denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)

    log_prob = logits - torch.log(denom)


    loss_i = -(log_prob * pos).sum(dim=1) / pos_cnt

    return loss_i.mean()
