# Data

This project expects custom PPE datasets in YOLO format. Large data files are intentionally ignored by git.

Expected structure:

```text
data/processed/
  dataset.yaml
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

Use `data/samples/` only for tiny demo assets that are safe to commit. Generated outputs belong in `data/outputs/` and are ignored.
