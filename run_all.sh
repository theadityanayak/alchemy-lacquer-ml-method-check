#!/usr/bin/env bash
# Rebuild results from data/ and run the checks. Fixed seed 42.
set -euo pipefail
cd "$(dirname "$0")"
python3 alchemy_lacquer_model.py
python3 test_model.py
