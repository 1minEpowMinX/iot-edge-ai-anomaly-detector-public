#!/usr/bin/env bash
# Steady "gateway duty cycle" background load for a realistic NORMAL baseline.
#
# CRITICAL: this represents the gateway's NORMAL state. Run it continuously
# during BOTH `collect` (baseline) AND `live` (inference). If it runs in only
# one of the two, the distributions differ and you get false positives.
#
# Requires: iputils-ping, coreutils (both present on a stock Ubuntu Server).
set -euo pipefail

PERIOD="${1:-5}"                        # seconds between duty cycles
LOG="${2:-$HOME/gateway_telemetry.log}"

echo "[gateway_load] duty cycle every ${PERIOD}s -> disk:${LOG} + loopback telemetry"
echo "[gateway_load] keep this running during BOTH collect and live (Ctrl-C to stop)"

while true; do
  # Sensor logging -> small disk write, flushed to disk (disk_write_bps).
  printf '%s poll temp=%d hum=%d\n' "$(date -Iseconds)" $((RANDOM % 50)) $((RANDOM % 100)) >> "$LOG"
  sync

  # Telemetry heartbeat -> a few packets on the loopback interface
  # (net_tx / net_packets_tx). No listener process needed.
  ping -c 2 -i 0.2 127.0.0.1 >/dev/null 2>&1 || true

  # Rotate the log so the disk does not slowly fill over a long demo.
  if [ "$(wc -l < "$LOG" 2>/dev/null || echo 0)" -gt 5000 ]; then
    : > "$LOG"
  fi

  sleep "$PERIOD"
done
