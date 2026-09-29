import os

# Пути к весам модели — форвардятся из docker-compose volumes
DINO_DIR = os.environ.get(
    "DINO_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "dinov3-vitl16"))
)
CKPT_PATH = os.environ.get(
    "CKPT_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "JDNFV_MASKED_FT_best_fp16.pt"))
)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "JDNFV_MASKED_FT")
DEVICE = os.environ.get("DEVICE", "auto")
_SIZE_STR = os.environ.get("SIZE", "auto")  # "auto" или "320"
SIZE: int | str = int(_SIZE_STR) if _SIZE_STR != "auto" else "auto"
YOLO_CKPT_PATH = os.environ.get(
    "YOLO_CKPT_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "yolo11s-seg.pt"))
)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8001"))
