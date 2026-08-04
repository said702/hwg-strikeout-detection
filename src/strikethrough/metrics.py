from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .constants import LABEL_CLEAN, LABEL_NAMES, LABEL_STRIKE
from .utils import ensure_dir


def compute_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = predictions["true_label"].astype(int)
    y_pred = predictions["pred_label"].astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[LABEL_CLEAN, LABEL_STRIKE],
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "label": [LABEL_CLEAN, LABEL_STRIKE],
            "label_name": [LABEL_NAMES[LABEL_CLEAN], LABEL_NAMES[LABEL_STRIKE]],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    overall = pd.DataFrame(
        [
            {
                "n_samples": int(len(predictions)),
                "macro_precision": float(precision.mean()),
                "macro_recall": float(recall.mean()),
                "macro_f1": float(f1.mean()),
                "struck_out_f1": float(per_class.loc[per_class["label"] == LABEL_STRIKE, "f1"].iloc[0]),
                "non_struck_out_f1": float(per_class.loc[per_class["label"] == LABEL_CLEAN, "f1"].iloc[0]),
            }
        ]
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[LABEL_CLEAN, LABEL_STRIKE])
    cm = pd.DataFrame(matrix, index=["true_non_struck_out", "true_struck_out"], columns=["pred_non_struck_out", "pred_struck_out"])
    return overall, per_class, cm


def compute_per_type_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strike_type, group in predictions.groupby("strike_type", dropna=False):
        overall, _, _ = compute_metrics(group)
        row = overall.iloc[0].to_dict()
        row["strike_type"] = strike_type
        rows.append(row)
    return pd.DataFrame(rows)


def save_metric_bundle(output_dir: Path, predictions: pd.DataFrame) -> None:
    ensure_dir(output_dir)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    overall, per_class, cm = compute_metrics(predictions)
    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    cm.to_csv(output_dir / "confusion_matrix.csv")
    compute_per_type_metrics(predictions).to_csv(output_dir / "per_type_metrics.csv", index=False)
