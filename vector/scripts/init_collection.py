#!/usr/bin/env python3
"""Создание коллекции галереи в Qdrant (идемпотентно, только стандартная библиотека).

Точка в коллекции = один размеченный автомобиль в кадре (ТЗ п.5: в кадре может быть несколько ТС):
  вектор   float32[EMBEDDING_DIM], метрика Cosine (Qdrant нормирует вектор при записи; score = косинусное сходство);
  payload  image_id (ключ кадра в S3), bbox [x, y, w, h], vehicle_id (идентичность ТС),
           batch_id (партия импорта).

Параметры берутся из переменных окружения, значения по умолчанию совпадают с vector/.env.example.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "gallery")
DIM = int(os.environ.get("EMBEDDING_DIM", "2048"))
QUANTIZATION = os.environ.get("QDRANT_QUANTIZATION", "none")  # none | scalar
HNSW_M = int(os.environ.get("HNSW_M", "16"))
HNSW_EF_CONSTRUCT = int(os.environ.get("HNSW_EF_CONSTRUCT", "128"))

# Поля payload, по которым backend фильтрует и группирует результаты
KEYWORD_INDEXES = ("image_id", "vehicle_id", "batch_id")


def call(method, path, body=None):
    """REST-вызов Qdrant. Возвращает (HTTP-статус, разобранный JSON)."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{URL}{path}", data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if API_KEY:
        request.add_header("api-key", API_KEY)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def wait_ready():
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"{URL}/readyz", timeout=5):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    sys.exit(f"Qdrant не отвечает по адресу {URL}")


def collection_config():
    config = {
        "vectors": {"size": DIM, "distance": "Cosine", "datatype": "float32"},
        "hnsw_config": {"m": HNSW_M, "ef_construct": HNSW_EF_CONSTRUCT},
        "on_disk_payload": True,
    }
    if QUANTIZATION == "scalar":
        # int8 в RAM для быстрого отбора кандидатов; исходные float32 остаются на диске для рескоринга
        config["quantization_config"] = {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}}
        config["vectors"]["on_disk"] = True
    elif QUANTIZATION != "none":
        sys.exit(f"QDRANT_QUANTIZATION должен быть none или scalar, получено '{QUANTIZATION}'")
    return config


def main():
    wait_ready()
    path = f"/collections/{urllib.parse.quote(COLLECTION)}"

    status, info = call("GET", path)
    if status == 200:
        size = info["result"]["config"]["params"]["vectors"]["size"]
        if size != DIM:
            sys.exit(
                f"Коллекция '{COLLECTION}' уже существует с размерностью {size}, а EMBEDDING_DIM={DIM}. "
                "Удалите коллекцию (или volume qdrant-data) и запустите инициализацию заново."
            )
        print(f"[init] коллекция '{COLLECTION}' уже есть (dim={size}), пропускаю создание")
    elif status == 404:
        status, reply = call("PUT", f"{path}?wait=true", collection_config())
        if status != 200:
            sys.exit(f"Не удалось создать коллекцию: {status} {reply}")
        print(f"[init] коллекция '{COLLECTION}' создана: dim={DIM}, Cosine, HNSW m={HNSW_M}, "
              f"ef_construct={HNSW_EF_CONSTRUCT}, quantization={QUANTIZATION}")
    else:
        sys.exit(f"Неожиданный ответ Qdrant при проверке коллекции: {status} {info}")

    for field in KEYWORD_INDEXES:
        status, reply = call("PUT", f"{path}/index?wait=true", {"field_name": field, "field_schema": "keyword"})
        if status != 200:
            sys.exit(f"Не удалось создать индекс по полю '{field}': {status} {reply}")
    print(f"[init] индексы payload готовы: {', '.join(KEYWORD_INDEXES)}")


if __name__ == "__main__":
    main()
