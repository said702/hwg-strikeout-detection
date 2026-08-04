from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from .constants import LABEL_CLEAN, LABEL_STRIKE
from .datasets import Sample


def class_counts(samples: Iterable[Sample]) -> Counter:
    return Counter(sample.label for sample in samples)


def class_weights(samples: list[Sample]) -> list[float]:
    counts = class_counts(samples)
    if counts.get(LABEL_CLEAN, 0) == 0 or counts.get(LABEL_STRIKE, 0) == 0:
        raise ValueError(f"Both classes are required for weighted loss, got {dict(counts)}")
    max_count = max(counts.values())
    return [max_count / counts[LABEL_CLEAN], max_count / counts[LABEL_STRIKE]]


def balanced_sample_weights(samples: list[Sample]) -> list[float]:
    counts = class_counts(samples)
    return [1.0 / max(counts[sample.label], 1) for sample in samples]


def stratified_split(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    rng = random.Random(seed)
    train, val = [], []
    for group in grouped.values():
        group = list(group)
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_ratio))) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def fixed_dev_test_split(
    samples: list[Sample], test_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Stratified-by-label split into (development_pool, fixed_test_set).

    The fixed test set is stratified by class label and deterministic for a
    given seed. It is used by ``learning_curve`` so that the same held-out
    test samples are never used for training or validation across all
    sample sizes and repetitions.
    """
    dev: list[Sample] = []
    test: list[Sample] = []
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    for group in grouped.values():
        group = list(group)
        rng.shuffle(group)
        n_test = int(round(len(group) * test_ratio))
        test.extend(group[:n_test])
        dev.extend(group[n_test:])
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def three_way_split_by_label(
    samples: list[Sample], ratios: tuple[float, float, float], seed: int
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Stratified-by-label three-way split into (train, val, test).

    Unlike ``three_way_split`` (which stratifies by label only), this keeps
    the split purely label-stratified and deterministic. Used for the
    single_type_training diagonal holdout.
    """
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    rng = random.Random(seed)
    train, val, test = [], [], []
    r_train, r_val, r_test = ratios
    total_ratio = r_train + r_val + r_test
    for group in grouped.values():
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * r_train / total_ratio))
        n_val = int(round(n * r_val / total_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_train += n_test
            n_test = 0
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val : n_train + n_val + n_test])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def stratified_three_way_kfold(
    samples: list[Sample], n_folds: int, val_ratio_of_train: float, seed: int
) -> list[tuple[list[Sample], list[Sample], list[Sample]]]:
    """Stratified k-fold producing (train, val, test) per fold.

    Each fold holds out ~1/n_folds of the data as the test set (stratified by
    label). The remaining samples are split into train and validation via a
    stratified split with ``val_ratio_of_train``. With n_folds=10 and
    val_ratio_of_train=0.2 this yields an effective 72/18/10 split.
    """
    from sklearn.model_selection import StratifiedKFold

    indices = np.arange(len(samples))
    labels = np.array([sample.label for sample in samples])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits: list[tuple[list[Sample], list[Sample], list[Sample]]] = []
    for train_val_idx, test_idx in skf.split(indices, labels):
        train_val = [samples[int(i)] for i in train_val_idx]
        test = [samples[int(i)] for i in test_idx]
        train, val = stratified_split(train_val, val_ratio=val_ratio_of_train, seed=seed)
        splits.append((train, val, test))
    return splits


def balanced_type_vs_clean(
    strike_samples: list[Sample],
    clean_samples: list[Sample],
    seed: int,
    replacement: bool = False,
) -> list[Sample]:
    rng = random.Random(seed)
    clean = list(clean_samples)
    rng.shuffle(clean)
    n = len(strike_samples)
    if n == 0:
        return []
    if replacement and len(clean) < n:
        selected: list[Sample] = []
        while len(selected) < n:
            selected.extend(clean)
        clean_selected = selected[:n]
    else:
        clean_selected = clean[:n]
    return list(strike_samples) + clean_selected


def subset_by_size(samples: list[Sample], n: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    per_class = max(1, n // max(len(grouped), 1))
    selected = []
    for group in grouped.values():
        group = list(group)
        rng.shuffle(group)
        selected.extend(group[:per_class])
    if len(selected) < n:
        rest = [sample for sample in samples if sample not in selected]
        rng.shuffle(rest)
        selected.extend(rest[: n - len(selected)])
    rng.shuffle(selected)
    return selected[:n]


def subset_by_size_with_meta(
    samples: list[Sample], n: int, seed: int
) -> tuple[list[Sample], dict]:
    """Like ``subset_by_size`` but also returns sampling metadata.

    Metadata fields:
      - ``sampling_strategy``: ``"stratified_by_label"``
      - ``n_strike`` / ``n_clean``: counts in the selected subset
      - ``class_ratio``: ``n_strike / n_clean`` (or 0.0 if a class is empty)
    """
    subset = subset_by_size(samples, n, seed)
    counts = Counter(sample.label for sample in subset)
    n_strike = int(counts.get(LABEL_STRIKE, 0))
    n_clean = int(counts.get(LABEL_CLEAN, 0))
    ratio = (n_strike / n_clean) if n_clean else 0.0
    return subset, {
        "sampling_strategy": "stratified_by_label",
        "n_strike": n_strike,
        "n_clean": n_clean,
        "class_ratio": float(ratio),
    }


def expand_sample_sizes(spec) -> list[int]:
    """Expand a ``sample_sizes`` config entry into an explicit list.

    Accepts either an explicit list of integers or a mapping with
    ``start``/``stop``/``step`` keys (yielding ``range(start, stop+1, step)``).
    """
    if spec is None:
        return []
    if isinstance(spec, dict):
        start = int(spec.get("start", 10))
        stop = int(spec.get("stop", 500))
        step = int(spec.get("step", 10))
        return list(range(start, stop + 1, step))
    return [int(x) for x in spec]


def bootstrap_seeds(seed: int, repetitions: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 2**31 - 1, size=repetitions)]


def select_clean_prefix(clean_samples: list[Sample], n: int, seed: int) -> list[Sample]:
    """Deterministically select ``n`` clean samples from ``clean_samples``.

    Uses a fresh RNG seeded with ``seed`` so the selection is reproducible and
    independent of any other shuffling performed with the same seed on a
    different pool.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    clean = list(clean_samples)
    rng.shuffle(clean)
    return clean[:n]


def three_way_split(
    samples: list[Sample],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Stratified 70/15/15 (or any ratios) split by label.

    Returns (train, val, test). Deterministic for a given seed.
    """
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    rng = random.Random(seed)
    train, val, test = [], [], []
    r_train, r_val, r_test = ratios
    total_ratio = r_train + r_val + r_test
    for group in grouped.values():
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * r_train / total_ratio))
        n_val = int(round(n * r_val / total_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_train += n_test
            n_test = 0
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val : n_train + n_val + n_test])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def stratified_kfold_splits(samples: list[Sample], n_folds: int, seed: int) -> list[tuple[list[Sample], list[Sample]]]:
    from sklearn.model_selection import StratifiedKFold

    indices = np.arange(len(samples))
    labels = np.array([sample.label for sample in samples])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits: list[tuple[list[Sample], list[Sample]]] = []
    for train_idx, test_idx in skf.split(indices, labels):
        train = [samples[int(i)] for i in train_idx]
        test = [samples[int(i)] for i in test_idx]
        splits.append((train, test))
    return splits
