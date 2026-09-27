"""Extract latent representations from a trained SupMoCo encoder.

Saves raw 512-dimensional encoder features after global average pooling,
before L2 normalization and the projection head.
"""

from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
# 1. Paths and settings
SEED = 2024

DATA_DIR = Path("data")
NPY_DIR = DATA_DIR / "images"
FILENAME_DIR = DATA_DIR / "filenames"
CHECKPOINT_PATH = Path("checkpoints") / f"SupMoCo_seed{SEED}" / "best_model.pt"
OUTPUT_DIR = Path("latent_vectors") / f"SupMoCo_seed{SEED}"

GROUPS = [
    "Control_GBA_C19",
    "Control_SNCA_C19",
    "Control_C4",
    "Control_C18",
    "SNCA",
    "GBA",
    "PINK1",
]

IMG_SIZE = 128
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

EXTRACT_BATCH_SIZE = 512
NUM_WORKERS = 0
USE_BF16 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

# 2. SupMoCo encoder
def conv2d(
    in_ch,
    out_ch,
    k=3,
    stride=1,
    padding=1,
    dilation=1,
    bias=True,
):
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
    def __init__(
        self,
        in_ch,
        out_ch,
        dilation=1,
    ):
        super().__init__()
        self.c1 = conv2d(
            in_ch,
            out_ch,
            3,
            1,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.c2 = conv2d(
            out_ch,
            out_ch,
            3,
            1,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.proj = None
        if in_ch != out_ch:
            self.proj = conv2d(
                in_ch,
                out_ch,
                1,
                1,
                padding=0,
                bias=False,
            )
    def forward(self, x):
        identity = x
        x = F.relu(
            x,
            inplace=True,
        )
        x = self.c1(x)
        x = F.relu(
            x,
            inplace=True,
        )
        x = self.c2(x)
        if self.proj is not None:
            identity = self.proj(
                identity
            )
        return x + identity
class Stage(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        n_blocks,
        dilation,
        use_ckpt=False,
        ckpt_segments=0,
    ):
        super().__init__()
        self.use_ckpt = bool(
            use_ckpt
        )
        self.ckpt_segments = int(
            ckpt_segments
        )
        blocks = [
            ResBlock(
                in_ch,
                out_ch,
                dilation=dilation,
            )
        ]
        for _ in range(
            n_blocks - 1
        ):
            blocks.append(
                ResBlock(
                    out_ch,
                    out_ch,
                    dilation=dilation,
                )
            )
        self.blocks = nn.Sequential(
            *blocks
        )
    def forward(self, x):
        # Checkpointing is irrelevant during inference.
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
            conv2d(
                3,
                64,
                k=3,
                stride=2,
                padding=1,
                bias=True,
            )
        )
        self.stage2 = Stage(
            64,
            128,
            b2,
            d2,
            False,
            1,
        )
        self.stage3 = Stage(
            128,
            256,
            b3,
            d3,
            False,
            1,
        )
        self.stage4 = Stage(
            256,
            512,
            b4,
            d4,
            True,
            ckpt_segments,
        )
        self.stage5 = Stage(
            512,
            OUT_DIM,
            b5,
            d5,
            True,
            ckpt_segments,
        )
        self.refine = Stage(
            OUT_DIM,
            OUT_DIM,
            int(refine_blocks),
            1,
            True,
            ckpt_segments,
        )
        self.trunk = nn.Sequential(
            self.stem,
            self.stage2,
            self.stage3,
            self.stage4,
            self.stage5,
            self.refine,
        )
        self.gap = nn.AdaptiveAvgPool2d(
            (1, 1)
        )
    def forward(self, x):
        x = self.trunk(x)
        return self.gap(
            x
        ).flatten(1)
def build_projector(
    in_dim,
    embed_dim,
    proj_layers,
    proj_hidden,
    use_bn,
    dropout,
):
    def lin(a, b):
        return nn.Linear(
            a,
            b,
            bias=False,
        )
    def bn(d):
        return (
            nn.BatchNorm1d(d)
            if use_bn
            else nn.Identity()
        )
    def do():
        return (
            nn.Dropout(dropout)
            if dropout > 0
            else nn.Identity()
        )
    if proj_layers <= 1:
        return nn.Sequential(
            lin(
                in_dim,
                embed_dim,
            )
        )
    if proj_layers == 2:
        return nn.Sequential(
            lin(
                in_dim,
                proj_hidden,
            ),                 # projector.0
            bn(proj_hidden),  # projector.1
            nn.ReLU(
                inplace=False
            ),                # projector.2
            do(),             # projector.3
            lin(
                proj_hidden,
                embed_dim,
            ),                 # projector.4
        )
    if proj_layers == 3:
        return nn.Sequential(
            lin(
                in_dim,
                proj_hidden,
            ),
            bn(proj_hidden),
            nn.ReLU(
                inplace=False
            ),
            do(),
            lin(
                proj_hidden,
                proj_hidden,
            ),
            bn(proj_hidden),
            nn.ReLU(
                inplace=False
            ),
            do(),
            lin(
                proj_hidden,
                embed_dim,
            ),
        )
    raise ValueError(
        f"Unsupported proj_layers={proj_layers}"
    )
class SupMoCoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.use_l2_norm_pool = (
            USE_L2_NORM_POOL
        )
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
            pooled = F.normalize(
                pooled,
                dim=1,
            )
        z = self.projector(
            pooled
        )
        return F.normalize(
            z,
            dim=1,
        )
# 3. Preprocessing
class SafeInstanceNormalize:
    def __init__(
        self,
        threshold=0.01,
    ):
        self.threshold = float(
            threshold
        )
    def __call__(
        self,
        tensor,
    ):
        mean = tensor.mean(
            dim=[1, 2],
            keepdim=True,
        )
        std = tensor.std(
            dim=[1, 2],
            keepdim=True,
        ).clamp_min(
            self.threshold
        )
        return (
            tensor - mean
        ) / std
class FullGroupDataset(Dataset):
    """
    Load all images from one source group.
    Extraction preprocessing:
    - uint16 -> float / 65535
    - HWC -> CHW
    - SafeInstanceNormalize
    - no augmentation
    """
    def __init__(
        self,
        group,
    ):
        self.group = group
        self.npy_path = NPY_DIR / f"{group}.npy"
        self.filename_path = FILENAME_DIR / f"{group}_filenames.npy"
        if not self.npy_path.is_file():
            raise FileNotFoundError(
                self.npy_path
            )
        if not self.filename_path.is_file():
            raise FileNotFoundError(
                self.filename_path
            )
        self.images = np.load(
            self.npy_path,
            mmap_mode="r",
        )
        self.filenames = np.load(
            self.filename_path,
            allow_pickle=True,
        )
        if len(
            self.images
        ) != len(
            self.filenames
        ):
            raise ValueError(
                f"{group}: "
                f"images={len(self.images):,}, "
                f"filenames={len(self.filenames):,}"
            )
        self.normalize = SafeInstanceNormalize(
            threshold=0.01
        )
    def __len__(self):
        return len(
            self.images
        )
    def __getitem__(
        self,
        index,
    ):
        image = self.images[
            index
        ]
        if image.dtype != np.uint16:
            raise ValueError(
                f"{self.group}[{index}]: "
                f"expected uint16, got {image.dtype}"
            )
        if image.shape != (
            IMG_SIZE,
            IMG_SIZE,
            3,
        ):
            raise ValueError(
                f"{self.group}[{index}]: "
                f"shape={image.shape}"
            )
        x = torch.from_numpy(
            np.asarray(
                image
            ).copy()
        )
        x = (
            x.permute(
                2,
                0,
                1,
            )
            .contiguous()
            .float()
            / 65535.0
        )
        x = self.normalize(
            x
        )
        return x
# 4. Load best checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)
checkpoint_epoch = int(
    checkpoint.get(
        "epoch",
        -1,
    )
)
checkpoint_val_acc = checkpoint.get(
    "val_acc",
    checkpoint.get(
        "best_acc",
        None,
    ),
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(
    "Checkpoint epoch:",
    checkpoint_epoch,
)
print(
    "Checkpoint validation accuracy:",
    checkpoint_val_acc,
)
print(
    "Output directory:",
    OUTPUT_DIR,
)
model = SupMoCoModel()
if "model_q" in checkpoint:
    load_result = model.load_state_dict(
        checkpoint["model_q"],
        strict=True,
    )
    print(
        "Loaded model_q:",
        load_result,
    )
elif "encoder_q" in checkpoint:
    load_result = model.encoder.load_state_dict(
        checkpoint["encoder_q"],
        strict=True,
    )
    print(
        "Loaded encoder_q:",
        load_result,
    )
else:
    raise KeyError(
        "Checkpoint contains neither model_q nor encoder_q. "
        f"Keys: {list(checkpoint.keys())}"
    )
model = (
    model.to(
        DEVICE
    )
    .to(
        memory_format=torch.channels_last
    )
)
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(
        False
    )
# 5. Verify representation dimension
with torch.inference_mode():
    dummy = torch.zeros(
        2,
        3,
        IMG_SIZE,
        IMG_SIZE,
        device=DEVICE,
    ).to(
        memory_format=torch.channels_last
    )
    dummy_latent = model.encoder(
        dummy
    )
print(
    "Raw encoder output shape:",
    tuple(
        dummy_latent.shape
    ),
)
if dummy_latent.shape[1] != OUT_DIM:
    raise RuntimeError(
        f"Expected {OUT_DIM} dimensions, "
        f"got {dummy_latent.shape[1]}"
    )
del dummy
del dummy_latent
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
# 6. Extract representations
@torch.inference_mode()
def extract_one_group(
    group,
):
    dataset = FullGroupDataset(
        group
    )
    loader = DataLoader(
        dataset,
        batch_size=(
            EXTRACT_BATCH_SIZE
        ),
        shuffle=False,
        num_workers=(
            NUM_WORKERS
        ),
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        drop_last=False,
    )
    use_bf16 = (
        USE_BF16
        and DEVICE.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    latent_batches = []
    for images in tqdm(
        loader,
        desc=f"Extract {group}",
    ):
        images = images.to(
            DEVICE,
            non_blocking=True,
        ).to(
            memory_format=torch.channels_last
        )
        with torch.amp.autocast(
            device_type=DEVICE.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            # Raw custom-encoder GAP feature:
            # before L2 normalization and projector.
            latent = model.encoder(
                images
            )
        latent_batches.append(
            latent.float().cpu().numpy()
        )
    latents = np.concatenate(
        latent_batches,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )
    filenames = np.asarray(
        dataset.filenames,
        dtype=object,
    )
    if len(latents) != len(filenames):
        raise RuntimeError(
            f"{group}: "
            f"latents={len(latents):,}, "
            f"filenames={len(filenames):,}"
        )
    latent_path = os.path.join(
        OUTPUT_DIR,
        f"{group}_latent_raw.npy",
    )
    filename_path = OUTPUT_DIR / f"{group}_filenames.npy"
    np.save(
        latent_path,
        latents,
    )
    np.save(
        filename_path,
        filenames,
        allow_pickle=True,
    )
    print(
        f"\nSaved {group}: "
        f"{latents.shape}"
    )
    summary = {
        "group": group,
        "n_images": int(
            len(latents)
        ),
        "feature_dim": int(
            latents.shape[1]
        ),
        "latent_path": str(latent_path),
        "filename_path": str(filename_path),
    }
    del dataset
    del loader
    del latent_batches
    del latents
    del filenames
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return summary
# 7. Extract all groups
summary_rows = []
for group in GROUPS:
    summary_rows.append(
        extract_one_group(
            group
        )
    )
# 8. Save summary
summary_df = pd.DataFrame(
    summary_rows
)
summary_csv = OUTPUT_DIR / "extraction_summary.csv"
summary_json = OUTPUT_DIR / "extraction_summary.json"
summary_df.to_csv(
    summary_csv,
    index=False,
)
with open(
    summary_json,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "seed": SEED,
            "checkpoint": str(CHECKPOINT_PATH),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_val_acc": (
                float(checkpoint_val_acc)
                if checkpoint_val_acc is not None
                else None
            ),
            "latent_definition": (
                "raw custom encoder GAP output "
                "before L2 normalization and projector"
            ),
            "groups": summary_rows,
        },
        file,
        ensure_ascii=False,
        indent=2,
    )
print(
    "\n" + "=" * 100
)
print(
    "LATENT EXTRACTION COMPLETE"
)
print(
    "=" * 100
)
print(
    summary_df[
        [
            "group",
            "n_images",
            "feature_dim",
        ]
    ].to_string(
        index=False
    )
)
print(
    f"\nOutput directory: {OUTPUT_DIR}"
)
print(
    f"Summary CSV: {summary_csv}"
)
print(
    f"Summary JSON: {summary_json}"
)