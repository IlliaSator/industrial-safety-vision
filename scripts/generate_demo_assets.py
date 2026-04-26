from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tiny synthetic demo assets.")
    parser.add_argument("--output-dir", default="docs/assets")
    return parser.parse_args()


def main() -> None:
    from industrial_safety_vision.demo.synthetic import write_demo_assets

    args = parse_args()
    print(json.dumps(write_demo_assets(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
