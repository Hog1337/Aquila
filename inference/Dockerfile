# Inference сервис: FastAPI + PyTorch + DINOv3 + FalconFT
# Python 3.12 (как у backend): закреплённые версии (numpy 2.5.x и др.) не ставятся на 3.10 из pytorch/pytorch:2.2.0.
# CUDA-библиотеки приходят с колёсами torch (nvidia-* пакеты), на хосте нужен только драйвер + nvidia-container-toolkit.
FROM python:3.12-slim

WORKDIR /app

# Системные зависимости: OpenCV (для масок) и curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Кэш pip живёт в кэш-маунте BuildKit, а не в слое образа: при смене requirements.txt колёса не качаются заново
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY app/ ./app/

# Директория для весов модели (монтируется снаружи или копируется при сборке)
RUN mkdir -p /app/weights

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
