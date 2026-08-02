from __future__ import annotations

import runpy
from pathlib import Path


def test_benchmark_peak_rss_is_available() -> None:
    script = Path(__file__).parents[1] / "scripts" / "benchmark_efficiency.py"

    namespace = runpy.run_path(str(script))

    assert namespace["_peak_rss_bytes"]() > 0
