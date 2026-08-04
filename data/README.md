# Data Layout

This repository does not track large datasets or generated crops.

Expected local layout after preparation:

```text
data/
  HWG-Dataset/
    HWG-Dataset/
      HWG-collected/
      HWG-SOW-labels/
      HWG-written/
      HWG-synthetic/
  SWS/
  HWG-SOW/          (only if SOW images are available)
```

Set local restricted paths in `configs/data_sources.yaml`:

- `datasets.sow.dataset_root`: local SOW folder received by request
- `datasets.iam.words_root`: extracted IAM `words/` directory from `data/words.tgz`

Prepared manifests are written under `results/_prepared/manifests/`.
