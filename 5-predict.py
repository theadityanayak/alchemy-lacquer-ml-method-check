#!/usr/bin/env python3
"""Predict PU lacquer scratch hardness for a new formulation.

Usage (normalized coordinates, all in [0,1]):
    python3 predict.py 0.7208 0.0036 0.9973 0.6596
Usage (source-scale values from the authors' unnorm sheet):
    python3 predict.py --source-scale 1.0946 0.0036 4.9667 18.18

Feature order: vernetzung hdi_ipdi mattierungsmittel pigmentpaste
Prints predicted Kratzhaerte, the calibrated observation std, and the 95%
interval. Predictions outside the tested region are flagged.
"""
import json
import sys
from pathlib import Path

import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
FEATS = ["vernetzung", "hdi_ipdi", "mattierungsmittel", "pigmentpaste"]


def main():
    args = sys.argv[1:]
    physical = "--source-scale" in args or "--physical" in args
    args = [a for a in args if a not in ("--source-scale", "--physical")]
    if len(args) != 4:
        print(__doc__)
        sys.exit(1)
    vals = [float(a) for a in args]

    card = json.load(open(HERE / "model_card.json"))
    try:
        gp = joblib.load(HERE / "model.joblib")
    except Exception as exc:  # version mismatch between sklearn builds
        sys.exit("Could not load model.joblib ({}). Install the pinned versions with pip install -r requirements.txt, "
                 "or retrain with python3 alchemy_lacquer_model.py".format(type(exc).__name__))
    inflate = card["inflation_factor"]

    if physical:
        norm = [(v - card["denorm"][f]["intercept"]) / card["denorm"][f]["slope"]
                for f, v in zip(FEATS, vals)]
    else:
        norm = vals

    x = np.array(norm, dtype=float).reshape(1, -1)
    warn = []
    for f, v, raw in zip(FEATS, norm, vals):
        if not (0.0 <= v <= 1.0):
            warn.append(f"{f}={raw} is outside the tested normalized range [0,1]: extrapolation")
    mean, std = gp.predict(x, return_std=True)
    std_cal = float(std[0]) * inflate
    print(f"input (normalized): " + ", ".join(f"{f}={v:.4f}" for f, v in zip(FEATS, norm)))
    print(f"predicted Kratzhaerte: {mean[0]:.3f}")
    print(f"calibrated observation std: {std_cal:.3f}")
    print(f"95% interval: [{mean[0] - 1.96 * std_cal:.3f}, {mean[0] + 1.96 * std_cal:.3f}]"
          f"  (half-width {1.96 * std_cal:.3f})")
    print(f"best measured value in the 42-row training set: {card['best_measured']:.2f}")
    for w in warn:
        print("WARNING: " + w)
    print("note: point prediction under real measurement noise; lab validation required.")


if __name__ == "__main__":
    main()
