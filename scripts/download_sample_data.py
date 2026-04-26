from __future__ import annotations

from pathlib import Path


def main() -> None:
    sample_dir = Path("data/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(
        "No large sample assets are downloaded by default. "
        "Place a small image or video in data/samples for local demos."
    )


if __name__ == "__main__":
    main()
