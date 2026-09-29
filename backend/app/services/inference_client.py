import json
import time

import httpx

from .. import config


class InferenceClient:
    def __init__(self):
        """Создаёт HTTP-клиент к сервису инференса. Принимает: без параметров."""
        self.url = config.INFERENCE_URL
        self.client = httpx.Client(timeout=120)

    def embed(self, image_bytes: bytes, x: int, y: int, w: int, h: int) -> tuple[list[float], int]:
        """Считает эмбеддинг объекта через сервис инференса.
        Принимает: image_bytes — байты кадра, x, y, w, h — bbox объекта."""
        resp = self.client.post(
            f"{self.url}/internal/embed",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"x": x, "y": y, "w": w, "h": h},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"], data["elapsed_ms"]

    def embed_batch_by_keys(self, specs: list[tuple[str, int, int, int, int]]) -> list[list[float]]:
        """Батч-инференс: вызывает /internal/embed-by-key-batch.
        Принимает список (image_id, x, y, w, h).
        Изображения читаются сервером с диска (/data/images/).
        V2 forward внутри сервера батчится → выше FPS.
        Возвращает список эмбеддингов в том же порядке."""
        payload = [{"image_id": iid, "x": x, "y": y, "w": w, "h": h} for iid, x, y, w, h in specs]
        resp = self.client.post(
            f"{self.url}/internal/embed-by-key-batch",
            data={"keys": json.dumps(payload)},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    def embed_batch_full(self, specs: list[tuple[str, int, int, int, int, str]]) -> list[list[float]]:
        """Батч-инференс с передачей изображений (base64).
        Принимает список (placeholder, x, y, w, h, image_base64).
        Сервер делает YOLO на всех + V2 батчем.
        Возвращает список эмбеддингов."""
        items = [
            {"image": img_b64, "x": x, "y": y, "w": w, "h": h}
            for _, x, y, w, h, img_b64 in specs
        ]
        resp = self.client.post(
            f"{self.url}/internal/embed-batch-full",
            content=json.dumps(items),
            headers={"content-type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]


inference = InferenceClient()
