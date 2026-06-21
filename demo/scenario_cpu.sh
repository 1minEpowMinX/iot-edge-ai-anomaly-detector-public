#!/usr/bin/env bash
# Demo Scenario #1: CPU spike + controlled fork storm.
#
# Safety: stress-ng is used with a LIMITED number of processes and
# a timeout, rather than the classic “:(){ :|:& };:” fork bomb, which would crash the VM.
#
# Requires: sudo apt-get install -y stress-ng
set -euo pipefail

DURATION="${1:-30}"   # seconds under load.
NCPU="$(nproc)"

echo "[scenario_cpu] let live warm up..."
sleep 5

echo "[scenario_cpu] CPU spike: ${NCPU} cores for ${DURATION}s."
stress-ng --cpu "${NCPU}" --timeout "${DURATION}s" --metrics-brief

echo "[scenario_cpu] fork-storm: 50 processes for 15 seconds (simulation of a fork bomb)."
stress-ng --fork 50 --timeout 15s --metrics-brief

echo "[scenario_cpu] the test scenario has been ended."
