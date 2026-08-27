import base64
import json
import os
import time
from pathlib import Path

import httpx


API_URL = os.getenv("API_URL", "http://localhost:8000")
IMAGES_DIR = Path("/client/images")
OUTPUT_DIR = Path("/client/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def encode_image(path: Path) -> str:
    """Lê um arquivo de imagem e retorna a string base64."""
    return base64.b64encode(path.read_bytes()).decode()


def wait_for_api(max_retries: int = 10, delay: float = 3.0) -> None:
    """Aguarda a API ficar disponível antes de enviar requisições."""
    for attempt in range(max_retries):
        try:
            response = httpx.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] API disponível | modelo: {data['model_name']}")
                return
        except httpx.ConnectError:
            pass

        print(f"[...] Aguardando API ({attempt + 1}/{max_retries})...")
        time.sleep(delay)

    raise RuntimeError("API não ficou disponível a tempo.")


def run_single_inference(image_path: Path, confidence: float = 0.25) -> None:
    """Envia uma imagem e salva a resposta estruturada da API."""
    print(f"\n─── Inferência: {image_path.name} ───")
    response = httpx.post(
        f"{API_URL}/predict",
        json={
            "image_base64": encode_image(image_path),
            "confidence": confidence,
            "model_name": "yolov8n.pt",
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    print(f"  Tempo de inferência : {data['inference_ms']} ms")
    print(f"  Resolução           : {data['image_width']}x{data['image_height']} px")
    print(f"  Detecções ({len(data['detections'])}):")
    for detection in data["detections"]:
        print(f"    • {detection['label']:20s} conf={detection['confidence']:.2f}")

    output_file = OUTPUT_DIR / f"{image_path.stem}_result.json"
    output_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Resultado salvo em  : {output_file}")


def run_batch_inference(image_paths: list[Path], confidence: float = 0.25) -> None:
    """Envia múltiplas imagens em uma única requisição."""
    print(f"\n─── Batch: {len(image_paths)} imagens ───")
    response = httpx.post(
        f"{API_URL}/predict/batch",
        json={
            "images_base64": [encode_image(path) for path in image_paths],
            "confidence": confidence,
            "model_name": "yolov8n.pt",
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    print(f"  Total batch         : {data['total_inference_ms']} ms")
    for index, result in enumerate(data["results"], start=1):
        print(
            f"  Imagem {index}: {len(result['detections'])} detecções "
            f"em {result['inference_ms']} ms"
        )


def main() -> None:
    wait_for_api()
    images = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))

    if not images:
        print("[AVISO] Nenhuma imagem encontrada em /client/images/")
    else:
        run_single_inference(images[0])
        if len(images) > 1:
            run_batch_inference(images)

    metrics = httpx.get(f"{API_URL}/metrics", timeout=5).json()
    print("\n─── Métricas da API ───")
    print(f"  Total de requisições : {metrics['total_requests']}")
    print(f"  Latência média       : {metrics['avg_inference_ms']} ms")


if __name__ == "__main__":
    main()
