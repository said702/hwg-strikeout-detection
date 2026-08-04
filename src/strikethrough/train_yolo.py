from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.nn.tasks import ClassificationModel

from .constants import LABEL_CLEAN, LABEL_NAMES, LABEL_STRIKE
from .datasets import Sample, stable_name
from .sampling import class_weights, stratified_split
from .utils import copy_or_link, ensure_dir


class WeightedClassificationLoss:
    def __init__(self, class_weights=None):
        self.class_weights = class_weights

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        target = batch["cls"].long()
        if self.class_weights is not None:
            loss = F.cross_entropy(
                preds.float(),
                target,
                weight=self.class_weights.to(device=preds.device, dtype=preds.dtype),
                reduction="mean",
            )
        else:
            loss = F.cross_entropy(preds.float(), target, reduction="mean")
        return loss, {"loss": loss.detach()}


class WeightedClassificationModel(ClassificationModel):
    def init_criterion(self):
        return WeightedClassificationLoss(getattr(self, "_class_weights", None))


class WeightedClassificationTrainer(ClassificationTrainer):
    _class_weights = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = WeightedClassificationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        model._class_weights = self._class_weights
        if weights:
            model.load(weights)
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model


def _materialize_yolo_dataset(train: list[Sample], val: list[Sample], root: Path, balance: bool) -> None:
    if root.exists():
        shutil.rmtree(root)
    for split_name, samples in [("train", train), ("val", val)]:
        if balance and split_name == "train":
            samples = _oversample_to_balance(samples)
        for sample in samples:
            class_dir = "non-struck-out" if sample.label == LABEL_CLEAN else "struck-out"
            dst = root / split_name / class_dir / stable_name(sample)
            copy_or_link(Path(sample.image_path), dst)


def _oversample_to_balance(samples: list[Sample]) -> list[Sample]:
    clean = [sample for sample in samples if sample.label == LABEL_CLEAN]
    strike = [sample for sample in samples if sample.label == LABEL_STRIKE]
    if not clean or not strike:
        return samples
    target = max(len(clean), len(strike))
    balanced = []
    for group in [clean, strike]:
        repeated = list(group)
        while len(repeated) < target:
            repeated.extend(group[: target - len(repeated)])
        balanced.extend(repeated[:target])
    return balanced


def train_yolo_classifier(
    train_samples: list[Sample],
    output_dir: Path,
    config: dict,
    seed: int,
    val_samples: list[Sample] | None = None,
) -> Path:
    from ultralytics import YOLO

    # Use weights as specified in the config (e.g. `yolo26n-cls.pt`).
    config.setdefault("model", {})

    weights = class_weights(train_samples) if config["training"].get("class_weighted_loss", True) else [1.0, 1.0]
    class_weight_tensor = torch.tensor(weights, dtype=torch.float32)

    WeightedClassificationTrainer._class_weights = class_weight_tensor

    if val_samples is None:
        val_ratio = float(config["training"].get("val_ratio", 0.2))
        train, val = stratified_split(train_samples, val_ratio=val_ratio, seed=seed)
    else:
        train, val = list(train_samples), list(val_samples)
    dataset_root = output_dir / "_yolo_dataset"
    _materialize_yolo_dataset(train, val, dataset_root, balance=config["training"].get("balanced_batches", True))

    model = YOLO(config["model"].get("weights", "yolo11n-cls.pt"))
    run_dir = ensure_dir(output_dir / "runs")
    model.train(
        data=str(dataset_root),
        epochs=int(config["training"].get("epochs", 50)),
        imgsz=int(config["model"].get("image_size", 224)),
        batch=int(config["training"].get("batch_size", 32)),
        workers=int(config["training"].get("workers", 4)),
        patience=int(config["training"].get("patience", 10)),
        project=str(run_dir),
        name="train",
        exist_ok=True,
        pretrained=bool(config["training"].get("pretrained", True)),
        trainer=WeightedClassificationTrainer,
        plots=False,
        save=True,
        seed=seed,
        verbose=False,
    )
    best = Path(model.trainer.best)
    if not best.exists():
        raise FileNotFoundError(f"YOLO best checkpoint not found: {best}")
    return best


def predict_yolo(checkpoint: Path, samples: list[Sample], config: dict) -> pd.DataFrame:
    from ultralytics import YOLO

    from .utils import ProgressReporter

    model = YOLO(str(checkpoint))
    image_size = int(config["model"].get("image_size", 224))
    pred_batch = int(config["training"].get("predict_batch_size", config["training"].get("batch_size", 32)))
    chunk_size = int(config["training"].get("predict_chunk_size", 1024))
    rows = []
    progress = ProgressReporter(len(samples))
    total = len(samples)
    offset = 0
    while offset < total:
        chunk = samples[offset : offset + chunk_size]
        chunk_paths = [str(s.image_path) for s in chunk]
        results = model.predict(chunk_paths, batch=pred_batch, imgsz=image_size, verbose=False, stream=True)
        for sample, result in zip(chunk, results):
            pred = int(result.probs.top1)
            conf = float(result.probs.top1conf)
            rows.append(
                {
                    "image_path": sample.image_path,
                    "dataset": sample.dataset,
                    "source_id": sample.source_id,
                    "strike_type": sample.strike_type,
                    "true_label": sample.label,
                    "true_label_name": LABEL_NAMES[sample.label],
                    "pred_label": pred,
                    "pred_label_name": LABEL_NAMES.get(pred, str(pred)),
                    "confidence": conf,
                }
            )
            progress.update(1)
        offset += len(chunk)
    progress.close()
    return pd.DataFrame(rows)
