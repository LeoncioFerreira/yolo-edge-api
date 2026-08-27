"""Quality gate that blocks deployment below the configured mAP@0.5."""

import argparse
from pathlib import Path

DEFAULT_THRESHOLD = 0.50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dataset", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[ERRO] Modelo não encontrado: {model_path}")
        return 1

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    dataset = args.dataset or "coco128.yaml"
    print(f"[INFO] Validando com dataset: {dataset}")
    metrics = model.val(data=dataset, split="val", verbose=False)
    map50 = float(metrics.box.map50)
    print(f"[INFO] mAP@0.5 = {map50:.4f}  |  Limiar: {args.threshold:.4f}")

    if map50 < args.threshold:
        print("[FALHA] mAP abaixo do limiar. Deploy bloqueado.")
        return 1

    print("[OK] Quality gate aprovado. Deploy autorizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
