#!/usr/bin/env bash
# Realistic "gateway duty cycle" background load for a NORMAL baseline.
#
# Emulates an industrial IoT gateway's steady-state behaviour:
#   * sensor logging       -> small periodic disk writes        (disk_write_bps)
#   * MQTT-style telemetry  -> a few PERSISTENT TCP connections to a local
#                             "broker" with a small JSON publish each cycle
#                             (periodic net_tx bursts + a steady non-zero tcp_conn)
#
# CRITICAL: this represents the gateway's NORMAL state. Run it continuously
# during BOTH `collect` (baseline) AND `live` (inference), or the distributions
# diverge and you get false positives:
#     nohup bash demo/gateway_load.sh >/dev/null 2>&1 &
#
# NOTE on tcp_conn: psutil needs privileges to count system-wide TCP. Run
# `collect`/`live` with sudo to actually capture these connections in tcp_conn;
# without sudo that channel falls back to ~0 (detection still works, but the
# realism point is lost).
#
# Self-contained: the "broker" is a tiny local discard server on 127.0.0.1.
#
# NOTE: no `set -e` — a transient publish failure must not kill the duty loop.
set -uo pipefail

PERIOD="${1:-5}"                          # seconds between duty cycles
LOG="${2:-$HOME/gateway_telemetry.log}"
BROKER_PORT="${3:-1883}"                   # MQTT default port (stand-in broker)
N_SENSORS=16                               # sensors aggregated per publish

cleanup() {
  exec 3>&- 2>/dev/null || true
  exec 4>&- 2>/dev/null || true
  [ -n "${SINK_PID:-}" ] && kill "$SINK_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- local stand-in broker: threaded TCP discard server on loopback ----------
python3 - "$BROKER_PORT" <<'PY' &
import socketserver, sys
class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            while self.request.recv(4096):
                pass
        except OSError:
            pass
class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
with Server(("127.0.0.1", int(sys.argv[1])), Handler) as s:
    s.serve_forever()
PY
SINK_PID=$!

# Wait for the broker to accept, then open the persistent "broker" connection.
for _ in $(seq 1 50); do
  if exec 3<>"/dev/tcp/127.0.0.1/${BROKER_PORT}" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
# Second persistent session (e.g. a Modbus-TCP slave link).
exec 4<>"/dev/tcp/127.0.0.1/${BROKER_PORT}" 2>/dev/null || true

echo "[gateway_load] duty cycle every ${PERIOD}s -> disk:${LOG} + 2 persistent TCP publishers (:${BROKER_PORT})"
echo "[gateway_load] keep this running during BOTH collect and live (Ctrl-C to stop)"
echo "[gateway_load] run collect/live with sudo so tcp_conn captures these connections"

while true; do
  ts="$(date -Iseconds)"

  # Build an MQTT-style payload: a batch of N_SENSORS aggregated readings.
  payload="{\"ts\":\"${ts}\",\"gw\":\"x86-industrial-gateway\",\"readings\":["
  for i in $(seq 1 "$N_SENSORS"); do
    payload+="{\"id\":${i},\"t\":$((RANDOM % 50)),\"h\":$((RANDOM % 100)),\"p\":$((900 + RANDOM % 200))},"
  done
  payload+="]}"

  # Sensor logging -> disk write, flushed to disk (disk_write_bps).
  printf '%s\n' "$payload" >> "$LOG"
  sync "$LOG" 2>/dev/null || sync

  # Telemetry publish over the persistent connections (net_tx burst; the
  # connections stay ESTABLISHED, keeping tcp_conn at a steady handful).
  printf '%s\n' "$payload" >&3 2>/dev/null || true
  printf '%s\n' "$payload" >&4 2>/dev/null || true

  # Rotate the log so the disk does not slowly fill over a long demo.
  if [ "$(wc -l < "$LOG" 2>/dev/null || echo 0)" -gt 5000 ]; then
    : > "$LOG"
  fi

  sleep "$PERIOD"
done
