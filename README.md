# Struck-Out Word Detection in Handwritten Text

This repository contains the code for the KONVENS 2026 paper

**“Struck-Out Word Detection in Handwritten Text: A Cross-Dataset Evaluation on the HWG Dataset”**

## Overview

This repository trains and evaluates supervised word-level image classifiers for detecting whether a handwritten word crop is **struck-out** or **non-struck-out**.

It reproduces the supervised experiments from the paper, including cross-dataset evaluation, intra-dataset cross-validation, strike-out type analysis, leave-one-type-out evaluation, single-type training, and learning curves.

The experiments use our newly created datasets with strike-out type annotations for nine categories: `single-horizontal`, `single-oblique`, `multiple-horizontal`, `multiple-oblique`, `crossed`, `circled`, `wavy`, `zigzag`, and `blackened`. 

The repository supports two image-classification backbones:

- **YOLO**: classification models from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
- **DINOv2**: Vision Transformer models with DINOv2 pretraining, loaded via [`timm`](https://github.com/huggingface/pytorch-image-models).

Experiments are configured through YAML files and write their outputs to the `results/` directory.

---

## Experiments

The supervised experiments are configured in:

```bash
configs/experiments.yaml
```

| Experiment | Short description |
|---|---|
| `cross_dataset` | Train on one dataset and test on all others. |
| `intra` | Stratified k-fold cross-validation within one dataset. |
| `strike_type_analysis` | Reuse or retrain cross-dataset models, depending on the YAML configuration, and report per-strike-type metrics across the configured datasets. |
| `leave_one_type_out` | Train on `HWG-written` using all strike types except one and test on the excluded type. |
| `single_type_training` | Train one model per strike type using strike-out samples and clean samples from `HWG-written`, then evaluate it on all nine target types using strike-out samples from `HWG-written` and clean samples from `HWG-collected`, producing a 9×9 matrix. |
| `learning_curve` | Train on increasing subsets (10, 20, …, 500) of `HWG-written` to measure sample efficiency, with evaluation on `HWG-collected`, `HWG-SOW`, and `SWS`. |


Each experiment produces per-sample predictions, per-class metrics, per-type metrics, confusion matrices, and aggregated summary files. A combined overview (`results/summary_combined.csv`) merges intra results for identical train/eval dataset pairs with cross_dataset results for differing pairs; cross-dataset self-evaluation is never substituted for intra.

## Data Setup

Most datasets are downloaded automatically on the first run.

| Dataset | Notes |
|---|---|
| `HWG-written` | Downloaded automatically from [Zenodo record 21560739](https://zenodo.org/records/21560739). |
| `HWG-synthetic` | Downloaded automatically from [Zenodo record 21560739](https://zenodo.org/records/21560739). |
| `HWG-collected` | Downloaded automatically from [Zenodo record 21560739](https://zenodo.org/records/21560739); the full version additionally requires the IAM `words/` folder. |
| `HWG-SOW` | Labels are downloaded automatically from [Zenodo record 21560739](https://zenodo.org/records/21560739); the original SOW images must be added manually. |
| `SWS` | Downloaded automatically from [Zenodo record 4765063](https://zenodo.org/records/4765063). |

Downloaded data is stored under:

```bash
data/
```

For setting up IAM and SOW paths, see [Optional Manual Data Setup](#optional-manual-data-setup).

---

## Installation

It is recommended to install the project inside a virtual environment.

### 1. Create and activate a virtual environment

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Upgrade Python packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch

PyTorch is **not** included in `requirements.txt`, because the correct build depends on your operating system, hardware, and CUDA version.

Use the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Example for an NVIDIA GPU with CUDA 12.8:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install the remaining dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Start

Run all experiments enabled in `configs/experiments.yaml`:

```bash
python main.py --config configs/experiments.yaml
```

On the first run, the redistributable HWG and SWS data are downloaded automatically and stored under `data/`.

Experiments with:

```yaml
enabled: true
```

are trained and evaluated. Results are written to:

```bash
results/
```

---

## Optional Manual Data Setup

Some source images cannot be redistributed and must be obtained separately.

If these datasets are missing, the code skips the corresponding parts and prints a warning at startup.

### IAM Handwriting Database

The IAM-derived portion of `HWG-collected` requires the original IAM word images.

1. Download `words.tgz` from the official IAM Handwriting Database page:

   https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database

2. Extract the archive to any local folder:

   ```bash
   tar -xzf words.tgz -C /path/to/your/choice/
   ```

3. Set `datasets.iam.words_root` in `configs/data_sources.yaml` to the extracted `words/` directory:

   ```yaml
   iam:
     words_root: /path/to/your/choice/words
   ```

Without this entry, only the redistributable subset of `HWG-collected` is available.

### HWG-SOW

`HWG-SOW` requires the original SOW document images, which are not redistributed with this repository.

The HWG-SOW annotations are downloaded automatically, but the original SOW images must be obtained separately. In our experiments, we requested the SOW images from the authors of the original SOW paper:

[Zhong et al., “Struck-out handwritten word detection and restoration for automatic descriptive answer evaluation”](https://www.sciencedirect.com/science/article/pii/S0923596524001152)

1. Obtain the SOW images from the original SOW authors.

2. Place them in any local folder.

3. Set `datasets.sow.dataset_root` in `configs/data_sources.yaml` to the SOW image folder:

   ```yaml
   sow:
     dataset_root: /path/to/your/SOW/

On the first run, the SOW word crops are generated automatically into:

```bash
data/HWG-SOW/
```

Only `dataset_root` needs to be set manually.

---

## Configuration

Experiments and model settings are configured through YAML files in `configs/`.

- `configs/experiments.yaml` controls which experiments are executed, the global seed, and the output directory.
- `configs/data_sources.yaml` configures dataset locations and manually required paths for IAM and SOW.
- `configs/yolo.yaml` contains YOLO-specific training settings.
- `configs/dino.yaml` contains DINOv2-specific training settings.

---

## Summary Files

Each enabled experiment writes its detailed metrics during execution, including
per-run files such as `overall_metrics.csv`, `per_class_metrics.csv`, and
`predictions.csv`.

After an experiment has completed successfully for all models enabled in that
experiment, its aggregated metrics are written to:


```bash
results/<experiment>/summary_metrics.csv
```

A global summary across all experiments is written to:

```bash
results/summary_all.csv
```

---

## Citation

If you use this code or the HWG dataset, please cite the paper:

```bibtex
@inproceedings{yasin2026struckout,
  title     = {Struck-Out Word Detection in Handwritten Text: A Cross-Dataset Evaluation on the HWG Dataset},
  author    = {Yasin, Said and Gold, Christian and Zesch, Torsten},
  booktitle = {Proceedings of the 22nd Conference on Natural Language Processing (KONVENS 2026)},
  year      = {2026},
  address   = {Hamburg, Germany}
}
```

Please also cite the original datasets where applicable, including IAM, SWS, and SOW, according to their respective licenses and citation requirements.

---

## License

This repository is provided for research and reproduction purposes.

The datasets retain their original licenses and access conditions:

- HWG via Zenodo
- SWS via Zenodo
- IAM via the IAM Handwriting Database terms
- SOW via the original SOW distribution terms
