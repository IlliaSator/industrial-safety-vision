from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()


DEFAULT_REPO_ID = "LibreYOLO/construction-safety-gsnvb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small YOLO PPE dataset for local training experiments."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output", default="data/raw/construction-safety-gsnvb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Install it with: python -m pip install huggingface_hub"
        ) from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output),
    )
    print(f"Dataset downloaded to {path}")
    print("Review the dataset license and split layout before training.")


if __name__ == "__main__":
    main()
