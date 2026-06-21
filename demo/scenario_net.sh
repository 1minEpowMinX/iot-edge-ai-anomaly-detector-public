#!/usr/bin/env bash
# Demo Scenario #2: Network Burst (net_tx/net_rx/tcp_conn).
#
# Requires: sudo apt-get install -y iperf3
set -euo pipefail

DURATION="${1:-30}"

echo "[scenario_net] let live warm up..."
sleep 5

# Local iperf3 server on the loopback interface.
iperf3 -s -1 -p 5201 >/dev/null 2>&1 &
SRV_PID=$!
sleep 1

echo "[scenario_net] Network spike: 16 parallel threads for ${DURATION}s."
# -P 16: Many TCP connections at once.
iperf3 -c 127.0.0.1 -p 5201 -P 16 -t "${DURATION}" || true

kill "${SRV_PID}" 2>/dev/null || true
echo "[scenario_net] the test scenario has been ended."
