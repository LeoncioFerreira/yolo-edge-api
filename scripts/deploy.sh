#!/bin/bash
set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-$HOME/yolo-edge-api}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
HEALTH_RETRIES="${HEALTH_RETRIES:-6}"
HEALTH_WAIT="${HEALTH_WAIT:-10}"

cd "$DEPLOY_PATH"

PREVIOUS=$(docker inspect yolo-api --format '{{.Config.Image}}' 2>/dev/null || echo "none")
echo "[INFO] Imagem atual: $PREVIOUS"
echo "[1/4] Baixando nova imagem..."
docker compose pull yolo-api
echo "[2/4] Iniciando nova versão..."
docker compose up -d yolo-api
echo "[3/4] Aguardando health check..."

SUCCESS=false
for attempt in $(seq 1 "$HEALTH_RETRIES"); do
    sleep "$HEALTH_WAIT"
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
        SUCCESS=true
        break
    fi
    echo "  Tentativa $attempt/$HEALTH_RETRIES falhou, aguardando..."
done

if [ "$SUCCESS" = true ]; then
    NEW=$(docker inspect yolo-api --format '{{.Config.Image}}')
    echo "[4/4] Health check OK"
    echo "[OK] Deploy bem-sucedido: $NEW"
    exit 0
fi

echo "[ERRO] Health check falhou."
if [ "$PREVIOUS" != "none" ]; then
    echo "[ROLLBACK] Revertendo para: $PREVIOUS"
    IMAGE="$PREVIOUS" docker compose up -d --force-recreate yolo-api
    echo "[ROLLBACK] Concluído."
else
    echo "[AVISO] Sem imagem anterior para rollback."
fi
exit 1
