#!/usr/bin/env python
"""Thin wrapper script to run inference: `python scripts/predict.py --input-csv ...`."""

from __future__ import annotations

from iris_mlops.scripts.predict import main

if __name__ == "__main__":
    main()
