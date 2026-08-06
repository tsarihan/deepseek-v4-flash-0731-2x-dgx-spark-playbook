#!/bin/bash
# Watchdog for deepseek-vllm.service.
#
# The unit is Type=oneshot + RemainAfterExit, so systemd cannot see an engine that
# died inside a container that is still up. This probes the API instead.
#
# Only acts when the unit is fully "active" (never during the ~8 min startup), and
# requires N consecutive failures so a single slow probe during a long prefill does
# not trigger a restart.

STATE=/run/deepseek-healthcheck.fails
THRESHOLD=3
# Generous: a 1M-token prefill can occupy the server for many minutes. We are
# probing /v1/models (cheap metadata), so this only fails if the engine is wedged.
PROBE_TIMEOUT=20

systemctl is-active --quiet deepseek-vllm.service || { rm -f "$STATE"; exit 0; }

if curl -fsS -m "$PROBE_TIMEOUT" http://127.0.0.1:8888/v1/models >/dev/null 2>&1; then
    rm -f "$STATE"
    exit 0
fi

fails=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$fails" > "$STATE"
echo "deepseek health probe failed ($fails/$THRESHOLD)" >&2

if [ "$fails" -ge "$THRESHOLD" ]; then
    echo "restarting deepseek-vllm.service after $fails consecutive failures" >&2
    rm -f "$STATE"
    systemctl restart deepseek-vllm.service
fi
