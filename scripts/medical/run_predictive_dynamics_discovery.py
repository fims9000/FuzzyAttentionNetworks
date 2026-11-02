#!/usr/bin/env python3
"""Run the frozen train/validation-only predictive-dynamics concept experiment."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conceptfan_discovery.experiment import main

if __name__ == "__main__":
    main()
