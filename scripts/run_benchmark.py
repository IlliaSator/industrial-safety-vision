from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

if __name__ == "__main__":
    from industrial_safety_vision.benchmarks.benchmark_inference import main

    main()
