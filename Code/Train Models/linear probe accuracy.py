"""Linear probe evaluation of frozen latent representations.

Five trained encoders are compared using the same class-balanced train/test
samples for each repeat.
"""

import gc
import os
from pathlib import Path
import re
import random
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


# 1. Paths
DATA_DIR = Path("data")
LATENT_DIR = Path("latent_vectors")
OUTPUT_DIR = Path("outputs") / "linear_probe"

MODEL_PATHS = {
    "ResNet18_seed42": LATENT_DIR / "ResNet18_seed42",
    "ResNet18_seed123": LATENT_DIR / "ResNet18_seed123",
    "SupConMoCo_seed42": LATENT_DIR / "SupConMoCo_seed42",
    "SupConMoCo_seed123": LATENT_DIR / "SupConMoCo_seed123",
    "SupConMoCo_seed2024": LATENT_DIR / "SupConMoCo_seed2024",
}

FILENAME_DIR = DATA_DIR / "filenames"
INDEX_DIR = DATA_DIR / "split_indices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Dataset structure
GROUPS = [
    'Control_GBA_C19',
    'Control_SNCA_C19',
    'Control_C4',
    'Control_C18',
    'GBA',
    'PINK1',
    'SNCA',
]

GROUP_TO_CLASS = {
    'Control_GBA_C19': 'Control',
    'Control_SNCA_C19': 'Control',
    'Control_C4': 'Control',
    'Control_C18': 'Control',
    'GBA': 'GBA',
    'PINK1': 'PINK1',
    'SNCA': 'SNCA',
}

CLASSES = ['Control', 'SNCA', 'GBA', 'PINK1']
CLASS_TO_LABEL = {
    'Control': 0,
    'SNCA': 1,
    'GBA': 2,
    'PINK1': 3,
}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}


# 3. Probe settings
RANDOM_SEED = 42

TRAIN_PER_CLASS_CAP = 3000

TEST_PER_CLASS_CAP = None

N_REPEATS = 10

USE_L2_NORMALIZATION = False

PROBE_EPOCHS = 100
PROBE_BATCH_SIZE = 256
PROBE_LR = 30.0
PROBE_MOMENTUM = 0.9
PROBE_WEIGHT_DECAY = 0.0
PROBE_MILESTONES = [60, 80]
PROBE_GAMMA = 0.1
PROBE_BIAS = False

NUM_WORKERS = 2
PIN_MEMORY = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# 4. Reproducibility utilities
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # A linear layer is simple, but these settings make repeated runs as
    # reproducible as reasonably possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# 5. General utilities
def normalize_filename(value):
    value = os.path.basename(str(value))
    value = re.sub(
        r'\.(tif|tiff|png|jpg|jpeg|npy)$',
        '',
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r'_mask$',
        '',
        value,
        flags=re.IGNORECASE,
    )
    return value


def canonical_id(group, filename):
        return f'{group}::{normalize_filename(filename)}'


def l2_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def find_existing_file(directory, candidates):
    for candidate in candidates:
        path = Path(directory) / candidate
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        f'File not found.\ndirectory={directory}\ncandidates={candidates}'
    )


def resolve_group_latent_files(latent_dir, group):
    latent_path = find_existing_file(
        latent_dir,
        [
            f'{group}_latent_raw.npy',
            f'{group}_latents.npy',
            f'{group}_latent.npy',
            f'{group}_latent_l2.npy',
        ],
    )
    filename_path = find_existing_file(
        latent_dir,
        [
            f'{group}_filenames.npy',
            f'{group}_filename.npy',
        ],
    )
    return latent_path, filename_path


# 6. Build grouped train/test membership
def build_split_membership():
    membership = {}

    for group in GROUPS:
        original_filename_path = FILENAME_DIR / f'{group}_filenames.npy'
        train_index_path = INDEX_DIR / f'{group}_train.npy'
        test_index_path = INDEX_DIR / f'{group}_test.npy'

        for path in [
            original_filename_path,
            train_index_path,
            test_index_path,
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(path)

        original_filenames = np.load(
            original_filename_path,
            allow_pickle=True,
        )
        train_indices = np.asarray(np.load(train_index_path), dtype=np.int64)
        test_indices = np.asarray(np.load(test_index_path), dtype=np.int64)

        train_indices = train_indices[
            (train_indices >= 0) & (train_indices < len(original_filenames))
        ]
        test_indices = test_indices[
            (test_indices >= 0) & (test_indices < len(original_filenames))
        ]

        normalized = np.array(
            [normalize_filename(name) for name in original_filenames],
            dtype=object,
        )

        train_set = set(normalized[train_indices].tolist())
        test_set = set(normalized[test_indices].tolist())

        overlap = train_set.intersection(test_set)
        if overlap:
            raise RuntimeError(
                f'{group}: train/test overlap={len(overlap)}'
            )

        membership[group] = {
            'train': train_set,
            'test': test_set,
        }

        print(
            f'{group:20s} | '
            f'train={len(train_set):,} | '
            f'test={len(test_set):,}'
        )

    return membership


split_membership = build_split_membership()


# 7. Common samples across models
def scan_model_available_ids(model_name, latent_dir):
    model_ids = {
        'train': defaultdict(set),
        'test': defaultdict(set),
    }
    audit_rows = []

    for group in GROUPS:
        latent_path, filename_path = resolve_group_latent_files(latent_dir, group)

        latent_memmap = np.load(latent_path, mmap_mode='r')
        latent_filenames = np.load(filename_path, allow_pickle=True)

        if len(latent_memmap) != len(latent_filenames):
            raise ValueError(
                f'{model_name}/{group}: '
                f'latent={len(latent_memmap):,}, '
                f'filenames={len(latent_filenames):,}'
            )

        normalized = np.array(
            [normalize_filename(name) for name in latent_filenames],
            dtype=object,
        )

        train_mask = np.fromiter(
            (name in split_membership[group]['train'] for name in normalized),
            dtype=bool,
            count=len(normalized),
        )
        test_mask = np.fromiter(
            (name in split_membership[group]['test'] for name in normalized),
            dtype=bool,
            count=len(normalized),
        )

        if np.any(train_mask & test_mask):
            raise RuntimeError(
                f'{model_name}/{group}: latent train/test overlap detected'
            )

        class_name = GROUP_TO_CLASS[group]

        train_names = normalized[train_mask]
        test_names = normalized[test_mask]

        model_ids['train'][class_name].update(
            canonical_id(group, name) for name in train_names
        )
        model_ids['test'][class_name].update(
            canonical_id(group, name) for name in test_names
        )

        audit_rows.append({
            'model': model_name,
            'group': group,
            'class': class_name,
            'latent_dim': int(latent_memmap.shape[1]),
            'latent_total': int(len(latent_memmap)),
            'matched_train': int(train_mask.sum()),
            'matched_test': int(test_mask.sum()),
            'unmatched_or_val': int(
                len(latent_memmap) - train_mask.sum() - test_mask.sum()
            ),
            'latent_file': Path(latent_path).name,
        })

        print(
            f'{model_name:22s} | {group:20s} | '
            f'train={train_mask.sum():,} | test={test_mask.sum():,} | '
            f'dim={latent_memmap.shape[1]}'
        )

        del latent_memmap
        del latent_filenames
        del normalized
        gc.collect()

    return model_ids, audit_rows


print('\n' + '=' * 110)
print('Scanning samples available in every latent model')
print('=' * 110)

all_model_ids = {}
audit_rows_all = []

for model_name, latent_dir in MODEL_PATHS.items():
    ids, audit_rows = scan_model_available_ids(model_name, latent_dir)
    all_model_ids[model_name] = ids
    audit_rows_all.extend(audit_rows)


common_ids = {
    'train': {},
    'test': {},
}

for split_name in ['train', 'test']:
    for class_name in CLASSES:
        sets = [
            all_model_ids[model_name][split_name][class_name]
            for model_name in MODEL_PATHS
        ]
        common_ids[split_name][class_name] = set.intersection(*sets)

        print(
            f'COMMON {split_name:5s} | {class_name:8s} | '
            f'{len(common_ids[split_name][class_name]):,}'
        )


# 8. Balanced repeated sampling
def balanced_count_from_sets(split_sets, cap=None):
    counts = [len(split_sets[class_name]) for class_name in CLASSES]

    if min(counts) <= 0:
        raise ValueError(f'At least one empty class: {counts}')

    count = min(counts)
    if cap is not None:
        count = min(count, int(cap))
    return int(count)


TRAIN_PER_CLASS = balanced_count_from_sets(
    common_ids['train'],
    TRAIN_PER_CLASS_CAP,
)
TEST_PER_CLASS = balanced_count_from_sets(
    common_ids['test'],
    TEST_PER_CLASS_CAP,
)

print('\nBalanced common sample counts')
print(f'  train/class = {TRAIN_PER_CLASS:,}')
print(f'  test/class  = {TEST_PER_CLASS:,}')


def sample_common_ids(split_sets, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = set()

    for class_name in CLASSES:
        # Sorting before random indexing makes the exact sample selection
        # independent of Python set iteration order.
        candidates = np.array(sorted(split_sets[class_name]), dtype=object)
        choice_idx = rng.choice(
            len(candidates),
            size=per_class,
            replace=False,
        )
        selected.update(candidates[choice_idx].tolist())

    return selected


repeat_samples = {}

for repeat in range(N_REPEATS):
    repeat_seed = RANDOM_SEED + repeat * 101

    repeat_samples[repeat] = {
        'train': sample_common_ids(
            common_ids['train'],
            TRAIN_PER_CLASS,
            repeat_seed,
        ),
        'test': sample_common_ids(
            common_ids['test'],
            TEST_PER_CLASS,
            repeat_seed + 1,
        ),
        'seed': repeat_seed,
    }


# 9. Latent loading
def load_selected_model_latents(latent_dir, selected_ids):
    split_X = {'train': [], 'test': []}
    split_y = {'train': [], 'test': []}

    for group in GROUPS:
        latent_path, filename_path = resolve_group_latent_files(latent_dir, group)

        latent_memmap = np.load(latent_path, mmap_mode='r')
        latent_filenames = np.load(filename_path, allow_pickle=True)

        normalized = np.array(
            [normalize_filename(name) for name in latent_filenames],
            dtype=object,
        )
        canonical = np.array(
            [canonical_id(group, name) for name in normalized],
            dtype=object,
        )

        class_name = GROUP_TO_CLASS[group]
        label = CLASS_TO_LABEL[class_name]

        for split_name in ['train', 'test']:
            chosen_set = selected_ids[split_name]
            mask = np.fromiter(
                (cid in chosen_set for cid in canonical),
                dtype=bool,
                count=len(canonical),
            )

            selected = np.asarray(latent_memmap[mask], dtype=np.float32)

            if USE_L2_NORMALIZATION:
                selected = l2_normalize(selected)

            split_X[split_name].append(selected)
            split_y[split_name].append(
                np.full(len(selected), label, dtype=np.int64)
            )

        del latent_memmap
        del latent_filenames
        del normalized
        del canonical
        gc.collect()

    result = {}

    for split_name in ['train', 'test']:
        X = np.concatenate(split_X[split_name], axis=0)
        y = np.concatenate(split_y[split_name], axis=0)

        expected = (
            TRAIN_PER_CLASS if split_name == 'train' else TEST_PER_CLASS
        )

        counts = {
            LABEL_TO_CLASS[label]: int(np.sum(y == label))
            for label in range(len(CLASSES))
        }

        for class_name in CLASSES:
            if counts[class_name] != expected:
                raise RuntimeError(
                    f'{split_name}/{class_name}: expected {expected}, '
                    f'loaded {counts[class_name]}'
                )

        result[split_name] = {'X': X, 'y': y}

    return result


# 10. PyTorch linear probe
class LinearProbe(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(
            in_features,
            num_classes,
            bias=PROBE_BIAS,
        )

    def forward(self, x):
        return self.fc(x)


def train_and_evaluate_probe(X_train, y_train, X_test, y_test, seed):
    seed_everything(seed)

    X_train_tensor = torch.from_numpy(
        np.ascontiguousarray(X_train, dtype=np.float32)
    )
    y_train_tensor = torch.from_numpy(
        np.ascontiguousarray(y_train, dtype=np.int64)
    )
    X_test_tensor = torch.from_numpy(
        np.ascontiguousarray(X_test, dtype=np.float32)
    )
    y_test_tensor = torch.from_numpy(
        np.ascontiguousarray(y_test, dtype=np.int64)
    )

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and DEVICE.type == 'cuda',
        generator=make_generator(seed),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and DEVICE.type == 'cuda',
        drop_last=False,
    )

    model = LinearProbe(
        in_features=X_train.shape[1],
        num_classes=len(CLASSES),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=PROBE_LR,
        momentum=PROBE_MOMENTUM,
        weight_decay=PROBE_WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=PROBE_MILESTONES,
        gamma=PROBE_GAMMA,
    )

    for epoch in range(PROBE_EPOCHS):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(yb)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_seen += len(yb)

        scheduler.step()

        if (
            epoch == 0
            or (epoch + 1) % 20 == 0
            or epoch + 1 == PROBE_EPOCHS
        ):
            train_loss = total_loss / max(total_seen, 1)
            train_acc = total_correct / max(total_seen, 1)
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'    epoch {epoch + 1:3d}/{PROBE_EPOCHS} | '
                f'loss={train_loss:.4f} | '
                f'train_acc={train_acc * 100:.2f}% | '
                f'lr={current_lr:g}'
            )

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()

            predictions.append(pred)
            targets.append(yb.numpy())

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)

    if model.fc.bias is not None:
        raise RuntimeError('Bias is unexpectedly enabled in the linear probe.')

    return prediction, target


# 11. Evaluation
metric_rows = []
recall_rows = []
confusion_rows = []

print('\n' + '=' * 110)
print(f'Device: {DEVICE}')
print(
    'Probe: '
    f'nn.Linear(bias={PROBE_BIAS}), '
    f'SGD(lr={PROBE_LR}, momentum={PROBE_MOMENTUM}, '
    f'weight_decay={PROBE_WEIGHT_DECAY}), '
    f'epochs={PROBE_EPOCHS}'
)
print(f'L2 normalization: {USE_L2_NORMALIZATION}')
print('=' * 110)

for model_name, latent_dir in MODEL_PATHS.items():
    print('\n' + '=' * 110)
    print(model_name)
    print('=' * 110)

    for repeat in range(N_REPEATS):
        repeat_seed = repeat_samples[repeat]['seed']

        print(
            f'\n  repeat {repeat + 1}/{N_REPEATS} | seed={repeat_seed}'
        )

        data = load_selected_model_latents(
            latent_dir,
            selected_ids={
                'train': repeat_samples[repeat]['train'],
                'test': repeat_samples[repeat]['test'],
            },
        )

        X_train = data['train']['X']
        y_train = data['train']['y']
        X_test = data['test']['X']
        y_test = data['test']['y']

        prediction, target = train_and_evaluate_probe(
            X_train,
            y_train,
            X_test,
            y_test,
            seed=repeat_seed,
        )

        acc = accuracy_score(target, prediction)
        bal_acc = balanced_accuracy_score(target, prediction)
        macro_f1 = f1_score(target, prediction, average='macro')

        recalls = recall_score(
            target,
            prediction,
            labels=list(range(len(CLASSES))),
            average=None,
            zero_division=0,
        )

        cm = confusion_matrix(
            target,
            prediction,
            labels=list(range(len(CLASSES))),
        )

        metric_rows.append({
            'model': model_name,
            'repeat': repeat + 1,
            'sampling_seed': repeat_seed,
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'macro_f1': float(macro_f1),
            'train_per_class': TRAIN_PER_CLASS,
            'test_per_class': TEST_PER_CLASS,
            'latent_dim': int(X_train.shape[1]),
            'l2_normalized': USE_L2_NORMALIZATION,
            'bias': PROBE_BIAS,
            'optimizer': 'SGD',
            'lr': PROBE_LR,
            'momentum': PROBE_MOMENTUM,
            'weight_decay': PROBE_WEIGHT_DECAY,
            'epochs': PROBE_EPOCHS,
        })

        for label, value in enumerate(recalls):
            recall_rows.append({
                'model': model_name,
                'repeat': repeat + 1,
                'class': LABEL_TO_CLASS[label],
                'recall': float(value),
            })

        for true_label in range(len(CLASSES)):
            for predicted_label in range(len(CLASSES)):
                confusion_rows.append({
                    'model': model_name,
                    'repeat': repeat + 1,
                    'true_class': LABEL_TO_CLASS[true_label],
                    'predicted_class': LABEL_TO_CLASS[predicted_label],
                    'count': int(cm[true_label, predicted_label]),
                })

        print(
            f'  RESULT | accuracy={acc * 100:.2f}% | '
            f'balanced_accuracy={bal_acc * 100:.2f}% | '
            f'macro-F1={macro_f1:.4f}'
        )

        del data
        del X_train, y_train, X_test, y_test
        del prediction, target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 12. Save results
metrics_df = pd.DataFrame(metric_rows)
recall_df = pd.DataFrame(recall_rows)
confusion_df = pd.DataFrame(confusion_rows)
audit_df = pd.DataFrame(audit_rows_all)

summary_df = (
    metrics_df
    .groupby('model', as_index=False)
    .agg(
        n_repeats=('accuracy', 'count'),
        accuracy_mean=('accuracy', 'mean'),
        accuracy_sd=('accuracy', 'std'),
        accuracy_median=('accuracy', 'median'),
        balanced_accuracy_mean=('balanced_accuracy', 'mean'),
        balanced_accuracy_sd=('balanced_accuracy', 'std'),
        macro_f1_mean=('macro_f1', 'mean'),
        macro_f1_sd=('macro_f1', 'std'),
        train_per_class=('train_per_class', 'first'),
        test_per_class=('test_per_class', 'first'),
        latent_dim=('latent_dim', 'first'),
    )
)

per_class_summary_df = (
    recall_df
    .groupby(['model', 'class'], as_index=False)
    .agg(
        recall_mean=('recall', 'mean'),
        recall_sd=('recall', 'std'),
        recall_median=('recall', 'median'),
    )
)

common_count_rows = []
for split_name in ['train', 'test']:
    for class_name in CLASSES:
        common_count_rows.append({
            'split': split_name,
            'class': class_name,
            'common_available': len(common_ids[split_name][class_name]),
            'used_per_repeat': (
                TRAIN_PER_CLASS if split_name == 'train' else TEST_PER_CLASS
            ),
        })
common_counts_df = pd.DataFrame(common_count_rows)

metrics_path = OUTPUT_DIR / 'linear_probe_repeated_metrics.csv'
summary_path = OUTPUT_DIR / 'linear_probe_model_summary.csv'
recall_path = OUTPUT_DIR / 'linear_probe_per_class_recall.csv'
recall_summary_path = OUTPUT_DIR / 'linear_probe_per_class_recall_summary.csv'
confusion_path = OUTPUT_DIR / 'linear_probe_confusion_counts.csv'
audit_path = OUTPUT_DIR / 'linear_probe_latent_split_audit.csv'
common_counts_path = OUTPUT_DIR / 'linear_probe_common_sample_counts.csv'

metrics_df.to_csv(metrics_path, index=False)
summary_df.to_csv(summary_path, index=False)
recall_df.to_csv(recall_path, index=False)
per_class_summary_df.to_csv(recall_summary_path, index=False)
confusion_df.to_csv(confusion_path, index=False)
audit_df.to_csv(audit_path, index=False)
common_counts_df.to_csv(common_counts_path, index=False)


# 13. Summary
print('\n' + '=' * 110)
print('PyTorch SGD linear-probe comparison finished')
print('=' * 110)

print('\n[Protocol]')
print(f'Device              : {DEVICE}')
print(f'L2 normalization    : {USE_L2_NORMALIZATION}')
print(f'Bias                 : {PROBE_BIAS}')
print(f'Epochs               : {PROBE_EPOCHS}')
print(f'Batch size           : {PROBE_BATCH_SIZE}')
print(f'Initial LR           : {PROBE_LR}')
print(f'Momentum             : {PROBE_MOMENTUM}')
print(f'Weight decay         : {PROBE_WEIGHT_DECAY}')
print(f'LR milestones        : {PROBE_MILESTONES}')
print(f'Repeats              : {N_REPEATS}')
print(f'Train/class/repeat   : {TRAIN_PER_CLASS:,}')
print(f'Test/class/repeat    : {TEST_PER_CLASS:,}')
print('Same samples/model   : YES')

print('\n[Model summary]')
display_summary = summary_df.copy()

for column in [
    'accuracy_mean',
    'accuracy_sd',
    'accuracy_median',
    'balanced_accuracy_mean',
    'balanced_accuracy_sd',
]:
    display_summary[column] = display_summary[column] * 100

print(
    display_summary[
        [
            'model',
            'accuracy_mean',
            'accuracy_sd',
            'accuracy_median',
            'balanced_accuracy_mean',
            'balanced_accuracy_sd',
            'macro_f1_mean',
            'macro_f1_sd',
            'latent_dim',
        ]
    ].to_string(index=False)
)

print('\nSaved directory:')
print(OUTPUT_DIR)

print('\nMain files:')
for path in [
    metrics_path,
    summary_path,
    recall_summary_path,
    audit_path,
    common_counts_path,
]:
    print(' -', path)
