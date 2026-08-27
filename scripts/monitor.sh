#!/bin/bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

echo "--- Status do Sistema ---"
if [ -r /sys/class/thermal/thermal_zone0/temp ]; then
    awk '{printf "Temperatura: %.1f °C\n", $1/1000}' /sys/class/thermal/thermal_zone0/temp
fi
free -h | awk '/^Mem:/ {print "RAM livre: " $4}'
echo "API status: $(curl -fsS "$API_URL/health" | jq -r '.status')"
echo "Requests totais: $(curl -fsS "$API_URL/metrics" | jq -r '.total_requests')"
echo "Latência média: $(curl -fsS "$API_URL/metrics" | jq -r '.avg_inference_ms') ms"
