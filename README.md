# YOLO Edge API

Projeto da Aula 3: práticas de MLOps e CI/CD para inferência YOLO em Raspberry Pi 5.

## Validação local

```bash
docker compose build
docker compose up -d yolo-api
docker compose run --rm yolo-client
```

Swagger UI: `http://<IP_DO_RASPBERRY>:8000/docs`.

Os pesos são gerenciados por DVC. No Raspberry Pi, execute `dvc pull` antes do primeiro deploy.

## Pipeline

O workflow `.github/workflows/edge-deploy.yml` executa lint, 14 testes, build ARM64, quality gate e deploy com rollback. Ele requer os secrets `TAILSCALE_AUTHKEY`, `RPI_HOST`, `RPI_USER`, `RPI_SSH_KEY` e `RPI_DEPLOY_PATH`.
