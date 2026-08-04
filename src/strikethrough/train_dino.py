from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from .constants import LABEL_NAMES, LABEL_STRIKE
from .datasets import Sample
from .sampling import balanced_sample_weights, class_weights, stratified_split
from .utils import ensure_dir


def training_batch_size(config: dict) -> int:
    """Return the DINO training batch size strictly from config.

    Reads ``config["training"]["batch_size"]`` without a silent fallback so
    that an accidental override (e.g. hard-coded 32) cannot hide a missing
    config value. ``configs/dino.yaml`` sets this to 8.
    """
    return int(config["training"]["batch_size"])


def predict_batch_size(config: dict) -> int:
    training = config.get("training", {})
    return int(training.get("predict_batch_size", training.get("batch_size", 32)))


def configured_image_size(config: dict) -> int:
    """Return the square DINO input size from ``configs/dino.yaml``.

    The value is intentionally required instead of falling back to timm's
    model default. This makes the configured image size the single source of
    truth for model creation, training, validation, and prediction.
    """
    image_size = int(config["model"]["image_size"])
    if image_size <= 0:
        raise ValueError(f"model.image_size must be positive, got {image_size}.")
    return image_size


def model_data_config(model, image_size: int) -> dict:
    """Resolve timm preprocessing and override its input size from config."""
    from timm.data import resolve_model_data_config

    data_cfg = dict(resolve_model_data_config(model))
    data_cfg["input_size"] = (3, image_size, image_size)
    return data_cfg


class ImageSampleDataset:
    def __init__(self, samples: list[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch

        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return self.transform(image), torch.tensor(sample.label, dtype=torch.long)


def train_dino_classifier(train_samples: list[Sample], output_dir: Path, config: dict, seed: int, val_samples: list[Sample] | None = None) -> Path:
    import timm
    import torch
    from sklearn.metrics import f1_score
    from timm.data import create_transform
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"].get("name", "vit_small_patch14_dinov2.lvd142m")
    image_size = configured_image_size(config)
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=2,
        img_size=image_size,
    )

    # train only the binary classification head.
    train_head_only = bool(config["training"].get("train_head_only", True))
    if train_head_only:
        for parameter in model.parameters():
            parameter.requires_grad = False

        classifier = model.get_classifier()
        if not isinstance(classifier, nn.Module):
            raise RuntimeError(
                f"Could not resolve a trainable classifier head for model {model_name!r}."
            )
        for parameter in classifier.parameters():
            parameter.requires_grad = True

    model = model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("DINO has no trainable parameters after configuring head-only training.")
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"        [DINO] train_head_only={train_head_only} "
        f"trainable_parameters={trainable_count}/{total_count}",
        flush=True,
    )

    data_cfg = model_data_config(model, image_size)
    transform = create_transform(**data_cfg, is_training=True)
    eval_transform = create_transform(**data_cfg, is_training=False)
    print(
        f"        [DINO] configured image_size={image_size} "
        f"transform_input_size={data_cfg['input_size']}",
        flush=True,
    )

    if val_samples is None:
        val_ratio = float(config["training"].get("val_ratio", 0.2))
        train, val = stratified_split(train_samples, val_ratio=val_ratio, seed=seed)
    else:
        train, val = list(train_samples), list(val_samples)
    train_ds = ImageSampleDataset(train, transform)
    val_ds = ImageSampleDataset(val, eval_transform)

    sampler = None
    shuffle = True
    if config["training"].get("balanced_batches", True):
        sampler = WeightedRandomSampler(balanced_sample_weights(train), num_samples=len(train), replacement=True)
        shuffle = False

    loader = DataLoader(
        train_ds,
        batch_size=training_batch_size(config),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(config["training"].get("workers", 4)),
    )
    val_loader = DataLoader(val_ds, batch_size=training_batch_size(config), shuffle=False)

    weight_tensor = None
    if config["training"].get("class_weighted_loss", True):
        weight_tensor = torch.tensor(class_weights(train), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["training"].get("use_amp", True)) and device.type == "cuda")
    best_f1 = -1.0
    best_path = ensure_dir(output_dir) / "best_dino.pt"
    epochs = int(config["training"].get("epochs", 30))
    verified_train_size = False
    for epoch in range(epochs):
        model.train()
        for images, labels in loader:
            if not verified_train_size:
                actual_size = tuple(images.shape[-2:])
                expected_size = (image_size, image_size)
                if actual_size != expected_size:
                    raise RuntimeError(
                        f"DINO training transform produced {actual_size}, "
                        f"expected {expected_size} from model.image_size."
                    )
                print(
                    f"        [DINO] verified training tensor shape={tuple(images.shape)}",
                    flush=True,
                )
                verified_train_size = True
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        correct = 0
        total = 0
        val_labels: list[int] = []
        val_predictions: list[int] = []
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(dim=1)
                correct += int((pred == labels).sum().item())
                total += int(labels.numel())
                val_labels.extend(labels.cpu().tolist())
                val_predictions.extend(pred.cpu().tolist())

        acc = correct / max(total, 1)
        val_f1 = (
            float(
                f1_score(
                    val_labels,
                    val_predictions,
                    pos_label=LABEL_STRIKE,
                    zero_division=0,
                )
            )
            if val_labels
            else 0.0
        )
        print(
            f"        [DINO] epoch {epoch+1}/{epochs} "
            f"val_acc={acc:.4f} val_f1={val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_name": model_name,
                    "image_size": image_size,
                    "state_dict": model.state_dict(),
                    "selection_metric": "validation_f1",
                    "best_validation_f1": best_f1,
                    "validation_accuracy_at_best_f1": acc,
                    "train_head_only": train_head_only,
                },
                best_path,
            )
            print(
                f"        [DINO] saved new best checkpoint by val_f1={best_f1:.4f}: {best_path}",
                flush=True,
            )
    return best_path


def predict_dino(checkpoint: Path, samples: list[Sample], config: dict) -> pd.DataFrame:
    import timm
    import torch
    from timm.data import create_transform
    from torch.utils.data import DataLoader

    from .utils import ProgressReporter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device)
    model_name = payload.get("model_name", config["model"].get("name"))
    image_size = int(payload.get("image_size", configured_image_size(config)))
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=2,
        img_size=image_size,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    data_cfg = model_data_config(model, image_size)
    transform = create_transform(**data_cfg, is_training=False)
    print(
        f"        [DINO] prediction image_size={image_size} "
        f"transform_input_size={data_cfg['input_size']}",
        flush=True,
    )
    dataset = ImageSampleDataset(samples, transform)
    pred_batch = predict_batch_size(config)
    loader = DataLoader(dataset, batch_size=pred_batch, shuffle=False)
    rows = []
    offset = 0
    progress = ProgressReporter(len(samples))
    verified_prediction_size = False
    with torch.no_grad():
        for images, _ in loader:
            if not verified_prediction_size:
                actual_size = tuple(images.shape[-2:])
                expected_size = (image_size, image_size)
                if actual_size != expected_size:
                    raise RuntimeError(
                        f"DINO prediction transform produced {actual_size}, "
                        f"expected {expected_size} from checkpoint/config."
                    )
                print(
                    f"        [DINO] verified prediction tensor shape={tuple(images.shape)}",
                    flush=True,
                )
                verified_prediction_size = True
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            preds = probs.argmax(dim=1).tolist()
            confs = probs.max(dim=1).values.tolist()
            for pred, conf in zip(preds, confs):
                sample = samples[offset]
                offset += 1
                rows.append(
                    {
                        "image_path": sample.image_path,
                        "dataset": sample.dataset,
                        "source_id": sample.source_id,
                        "strike_type": sample.strike_type,
                        "true_label": sample.label,
                        "true_label_name": LABEL_NAMES[sample.label],
                        "pred_label": int(pred),
                        "pred_label_name": LABEL_NAMES.get(int(pred), str(pred)),
                        "confidence": float(conf),
                    }
                )
            progress.update(len(images))
    progress.close()
    return pd.DataFrame(rows)
