"""Encoder and supervised MoCo loss used for ResNet18 experiments.
Random seeds: 42, 123.
"""

from typing import List, Tuple

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.checkpoint import checkpoint_sequential

from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from torchvision.models import resnet18

SEED = 42

OUT_DIM = 512

EMBED_DIM = 512

PROJ_LAYERS = 2

PROJ_HIDDEN = 2048

PROJ_BN = False

PROJ_DROPOUT = 0.0

USE_L2_NORM_POOL = True

TEMPERATURE = 0.07

MOCO_M = 0.995

QUEUE_SIZE = 65536

class ResNet18Encoder(nn.Module):
    """
    Completely standard torchvision ResNet-18:
      conv1: 7x7, stride 2, padding 3
      maxpool: 3x3, stride 2
      layer1-4: standard BasicBlock configuration [2,2,2,2]
      adaptive average pooling
      fc replaced with Identity

    No custom stem, no SafeInstanceNorm inside the model, no architectural
    modifications. Preprocessing is handled only by the dataset.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()

        # weights=None keeps the comparison purely self-supervised from scratch.
        # Set PRETRAINED_IMAGENET=True only for a separate pretrained ablation.
        weights = "DEFAULT" if pretrained else None

        if pretrained:
            try:
                from torchvision.models import ResNet18_Weights
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception:
                backbone = resnet18(pretrained=True)
        else:
            backbone = resnet18(weights=None)

        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

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
        self.encoder = ResNet18Encoder(PRETRAINED_IMAGENET)
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
