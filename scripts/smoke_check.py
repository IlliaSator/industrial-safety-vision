from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-command local smoke check for the project."
    )
    parser.add_argument(
        "--skip-lint",
        action="store_true",
        help="Skip ruff check if you only want runtime validation.",
    )
    parser.add_argument(
        "--skip-demos",
        action="store_true",
        help="Skip image/video demo generation.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip mock benchmark generation.",
    )
    return parser.parse_args()


def build_steps(args: argparse.Namespace) -> list[Step]:
    python = sys.executable
    steps = [
        Step("Compile Python files", [python, "-m", "compileall", "src", "tests", "scripts"]),
        Step("Run unit tests", [python, "-m", "pytest", "-q"]),
    ]

    if not args.skip_lint:
        steps.append(Step("Run lint check", [python, "-m", "ruff", "check", "."]))

    if not args.skip_demos:
        steps.extend(
            [
                Step("Generate demo assets", [python, "scripts/generate_demo_assets.py"]),
                Step(
                    "Run mock image demo",
                    [
                        python,
                        "scripts/run_image_demo.py",
                        "--image",
                        "docs/assets/demo_input.jpg",
                        "--output",
                        "data/outputs",
                        "--mock",
                    ],
                ),
                Step(
                    "Run synthetic video demo",
                    [
                        python,
                        "scripts/run_video_demo.py",
                        "--synthetic",
                        "--output",
                        "data/outputs/synthetic_annotated.gif",
                        "--max-frames",
                        "30",
                    ],
                ),
            ]
        )

    if not args.skip_benchmark:
        steps.append(
            Step(
                "Run mock pipeline benchmark",
                [
                    python,
                    "scripts/run_benchmark.py",
                    "--mock",
                    "--output",
                    "reports/benchmark_results.json",
                    "--markdown-report",
                    "reports/benchmark_report_smoke.md",
                ],
            )
        )

    return steps


def run_step(step: Step) -> None:
    print(f"\n==> {step.name}", flush=True)
    print("$ " + " ".join(step.command), flush=True)
    subprocess.run(step.command, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != repo_root:
        print(f"Changing working directory to {repo_root}", flush=True)
        os.chdir(repo_root)
    args = parse_args()
    steps = build_steps(args)

    for step in steps:
        run_step(step)

    print("\nSmoke check completed successfully.", flush=True)
    if not args.skip_demos or not args.skip_benchmark:
        print(
            "Generated demo and benchmark artifacts are written under data/outputs/ and reports/.",
            flush=True,
        )


if __name__ == "__main__":
    main()
