import io
import time
from pathlib import Path
from contextlib import asynccontextmanager

from typing import Annotated
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from PIL import Image

from . import config
from .reid_embed import load_model
from .yolo_seg import apply_masking_gpu

model = None
model_error: str | None = None
model_version: str = "unknown"
yolo_model = None
yolo_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_error, model_version, yolo_model, yolo_error
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    ckpt_path = Path(config.CKPT_PATH)
    dino_dir = Path(config.DINO_DIR)
    print(f"[inference] loading model: ckpt={ckpt_path}, dino={dino_dir}, device={config.DEVICE}")

    if not ckpt_path.exists():
        model_error = f"чекпоинт не найден: {ckpt_path}"
        print(f"[inference] WARNING: {model_error}")
    elif not dino_dir.exists():
        model_error = f"DINOv3 директория не найдена: {dino_dir}"
        print(f"[inference] WARNING: {model_error}")
    else:
        try:
            t0 = time.time()
            model = load_model(str(dino_dir), str(ckpt_path), device=config.DEVICE, size=config.SIZE)
            model_version = config.MODEL_VERSION
            print(f"[inference] model loaded in {time.time() - t0:.1f}s, device={model.device}")
            # Компиляция backbone для ускорения (1.3-1.5x)
            if hasattr(model, 'compile_backbone'):
                model.compile_backbone(mode="default")
            # Прогрев: несколько forward-проходов для CUDA kernel init
            print(f"[inference] warming up...", flush=True)
            try:
                warm_pil = Image.new("RGB", (config.SIZE if isinstance(config.SIZE, int) else 320, config.SIZE if isinstance(config.SIZE, int) else 320), (128, 128, 128))
                for _ in range(3):
                    model.embed_pils([warm_pil], batch=1)
                print(f"[inference] warmup done in {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[inference] warmup error (non-fatal): {e}", flush=True)
        except Exception as e:
            model_error = str(e)
            print(f"[inference] ERROR loading model: {e}")

    # Загрузка YOLO11s-seg
    yolo_path = Path(config.YOLO_CKPT_PATH)
    print(f"[inference] loading YOLO: {yolo_path}")
    if yolo_path.exists():
        try:
            from ultralytics import YOLO
            t0 = time.time()
            yolo_model = YOLO(str(yolo_path))
            print(f"[inference] YOLO loaded in {time.time() - t0:.1f}s")
            # Прогрев YOLO: маленькая картинка, чтобы вызвать компиляцию кода
            try:
                import numpy as np
                warm_img = np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8)
                yolo_model(warm_img, verbose=False)
                print(f"[inference] YOLO warmup done")
            except Exception as ye:
                print(f"[inference] YOLO warmup error (non-fatal): {ye}")
        except Exception as e:
            yolo_error = str(e)
            print(f"[inference] ERROR loading YOLO: {e}")
    else:
        yolo_error = f"YOLO чекпоинт не найден: {yolo_path}"
        print(f"[inference] WARNING: {yolo_error}")

    yield
    if hasattr(model, 'close'):
        model.close()
    model = None
    yolo_model = None


app = FastAPI(title="Falcon ReID Inference", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if model else "degraded",
        "device": model.device if model else None,
        "model_error": model_error,
        "model_version": model_version,
        "yolo_loaded": yolo_model is not None,
        "yolo_error": yolo_error,
    }


@app.get("/internal/status")
async def internal_status():
    return {
        "model_version": model_version,
        "embedding_dim": getattr(model, "proj_dim", 512),
        "device": model.device if model else None,
        "dtype": "bf16" if model and (getattr(model, "use_half", False) or getattr(model, "use_bf16", False)) else "fp32",
        "model_loaded": model is not None,
        "model_error": model_error,
        "yolo_loaded": yolo_model is not None,
        "yolo_error": yolo_error,
    }


@app.post("/internal/embed")
async def embed(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
):
    """Embedding с маскированием чужих машин через YOLO11s-seg."""
    if model is None:
        raise HTTPException(503, f"Модель не загружена: {model_error or 'неизвестная ошибка'}")
    if yolo_model is None:
        raise HTTPException(503, f"YOLO не загружен: {yolo_error or 'неизвестная ошибка'}")

    t0 = time.time()
    contents = await image.read()
    if len(contents) == 0:
        raise HTTPException(422, "Пустое изображение")

    import torchvision.io
    raw = torch.frombuffer(bytearray(contents), dtype=torch.uint8)
    img_tensor = torchvision.io.decode_jpeg(raw, device='cuda')  # (3, H, W) uint8, сразу на GPU
    _, H, W = img_tensor.shape

    # Валидация bbox
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise HTTPException(422, f"Некорректный bbox: ({x},{y},{w},{h})")
    if x + w > W or y + h > H:
        raise HTTPException(422, f"BBox выходит за кадр: bbox({x},{y},{w},{h}) vs img({W},{H})")

    try:
        # YOLO на GPU: resize → normalize → batch dim → forward (без CPU round-trip)
        yolo_input = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0).float(), size=(640, 640),
            mode='bilinear', align_corners=False
        ).div_(255.0)
        yolo_results = yolo_model(yolo_input, verbose=False)[0]
    except Exception as e:
        import traceback
        print(f"[inference] YOLO error: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(500, f"YOLO error: {e}")

    try:
        masked_crop, iou_best = apply_masking_gpu(img_tensor, (x, y, w, h), yolo_results)
    except Exception as e:
        import traceback
        print(f"[inference] masking error: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(500, f"masking error: {e}")

    try:
        emb = model.embed_pils([masked_crop], batch=1)
    except Exception as e:
        import traceback
        print(f"[inference] DINOv3 forward error: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(500, f"DINOv3 error: {e}")

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "embedding": emb[0].tolist(),
        "elapsed_ms": elapsed_ms,
        "iou_best": iou_best,
        "masked": True,
        "yolo_detections": len(yolo_results) if yolo_results else 0,
    }


@app.post("/internal/embed-batch")
async def embed_batch(images: Annotated[list[UploadFile], File(description="Multiple image files")]):
    """Батчовый embed: принимает несколько изображений (multipart с одинаковым именем поля).
    Возвращает эмбеддинги для всех изображений сразу (GPU-батчинг).
    ВНИМАНИЕ: без YOLO маскирования — чистый V2 forward на переданных кропах.
    """
    if model is None:
        raise HTTPException(503, f"Модель не загружена: {model_error or 'неизвестная ошибка'}")

    t0 = time.time()
    pils = []
    for img_file in images:
        contents = await img_file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        pils.append(pil)

    embs = model.embed_pils(pils, batch=min(64, len(pils)))
    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "embeddings": embs.tolist(),
        "elapsed_ms": elapsed_ms,
        "n": len(embs),
    }


@app.post("/internal/embed-batch-full")
async def embed_batch_full(request: Request):
    """Полный батч-пайплайн: JSON body с массивом {image_base64, x, y, w, h}.
    Декодирует изображения, YOLO на всех одним батчем, маскирование, V2 батчем.
    """
    if model is None:
        raise HTTPException(503, f"Модель не загружена: {model_error or 'неизвестная ошибка'}")
    if yolo_model is None:
        raise HTTPException(503, f"YOLO не загружен: {yolo_error or 'неизвестная ошибка'}")

    import json
    import base64
    import torchvision.io
    t0 = time.time()
    body = await request.body()
    items = json.loads(body)

    # 1. Декодируем все изображения на GPU
    img_tensors = []
    bboxes = []
    for item in items:
        raw_bytes = base64.b64decode(item["image"])
        raw = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
        img = torchvision.io.decode_jpeg(raw, device='cuda')  # (3, H, W) uint8
        img_tensors.append(img)
        bboxes.append((item["x"], item["y"], item["w"], item["h"]))

    # 2. YOLO на всех одним батчем (FP16)
    t1 = time.time()
    yolo_tensors = []
    for t in img_tensors:
        t_r = torch.nn.functional.interpolate(
            t.unsqueeze(0).float(), size=(640, 640),
            mode='bilinear', align_corners=False
        ).squeeze(0).div_(255.0)
        yolo_tensors.append(t_r)
    yolo_batch = torch.stack(yolo_tensors)
    all_yolo_results = yolo_model(yolo_batch, verbose=False)
    t_yolo = time.time() - t1

    # 3. Маскирование + crop (на GPU, без PIL)
    t1 = time.time()
    crop_tensors = []
    for img_t, yres, (x, y, w, h) in zip(img_tensors, all_yolo_results, bboxes):
        crop_t, _ = apply_masking_gpu(img_t, (x, y, w, h), yres, return_tensor=True)
        crop_tensors.append(crop_t)
    t_mask = time.time() - t1

    del img_tensors

    # 4. V2 forward: препроцессинг + embed_tensor на GPU (без CPU round-trip)
    t1 = time.time()
    model_size = getattr(model, 'size', 320)
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda', dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda', dtype=torch.float32).view(1, 3, 1, 1)
    
    # Стек + resize (crop_tensors из return_tensor=True: 0-255, разные размеры)
    resized = torch.stack([
        torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(model_size, model_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        for t in crop_tensors
    ])
    pv = (resized / 255.0 - mean) / std  # нормализация ImageNet
    
    e = model.embed_tensor(pv)
    import numpy as np
    embs_np = e.cpu().numpy().astype(np.float32)
    embs = embs_np.tolist()
    t_v2 = time.time() - t1

    elapsed_ms = int((time.time() - t0) * 1000)
    return {"embeddings": embs, "elapsed_ms": elapsed_ms, "n": len(embs)}


@app.post("/internal/embed-by-key")
async def embed_by_key(image_id: str = Form(...), x: int = Form(...), y: int = Form(...),
                        w: int = Form(...), h: int = Form(...)):
    """Embedding с маскированием по image_id + bbox."""
    if model is None:
        raise HTTPException(503, f"Модель не загружена: {model_error or 'неизвестная ошибка'}")
    if yolo_model is None:
        raise HTTPException(503, f"YOLO не загружен: {yolo_error or 'неизвестная ошибка'}")

    t0 = time.time()
    img_path = Path("/data/images") / f"{image_id}.jpg"
    if not img_path.exists():
        raise HTTPException(404, f"Изображение не найдено: {img_path}")

    import torchvision.io
    raw = torch.frombuffer(bytearray(Path(img_path).read_bytes()), dtype=torch.uint8)
    img_tensor = torchvision.io.decode_jpeg(raw, device='cuda')
    yolo_input = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0).float(), size=(640, 640),
        mode='bilinear', align_corners=False
    ).div_(255.0)
    yolo_results = yolo_model(yolo_input, verbose=False)[0]
    masked_crop, iou_best = apply_masking_gpu(img_tensor, (x, y, w, h), yolo_results)
    emb = model.embed_pils([masked_crop], batch=1)
    elapsed_ms = int((time.time() - t0) * 1000)
    return {"embedding": emb[0].tolist(), "elapsed_ms": elapsed_ms, "iou_best": iou_best}


@app.post("/internal/embed-by-key-batch")
async def embed_by_key_batch(keys: str = Form(...)):
    """Батч: JSON со списком {image_id, x, y, w, h}.
    GPU JPEG decode → YOLO seg на ВСЕХ кадрах одним батчем →
    маскирование → V2 forward на всех crops одним батчем.
    """
    if model is None:
        raise HTTPException(503, f"Модель не загружена: {model_error or 'неизвестная ошибка'}")
    if yolo_model is None:
        raise HTTPException(503, f"YOLO не загружен: {yolo_error or 'неизвестная ошибка'}")

    import json
    import torchvision.io
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.time()
    specs = json.loads(keys)

    # 1. Загрузка изображений на GPU (nvJPEG decode, без CPU round-trip)
    img_tensors = [None] * len(specs)
    def load_one(idx_s):
        idx, s = idx_s
        iid = s["image_id"]
        p = Path("/data/images") / f"{iid}.jpg"
        if not p.exists():
            raise HTTPException(404, f"Изображение не найдено: {iid}")
        raw = torch.frombuffer(bytearray(p.read_bytes()), dtype=torch.uint8)
        img = torchvision.io.decode_jpeg(raw, device='cuda')  # (3, H, W) uint8, сразу на GPU
        return idx, img

    with ThreadPoolExecutor(max_workers=min(8, len(specs))) as pool:
        futures = [pool.submit(load_one, (i, s)) for i, s in enumerate(specs)]
        for f in as_completed(futures):
            idx, tensor = f.result()
            img_tensors[idx] = tensor

    # 2. YOLO на всех изображениях одним батчем + сохраняем resized_640 для fast masking
    t1 = time.time()
    yolo_imgsz = 640
    yolo_tensors = []  # resized_640 для каждого изображения
    for t in img_tensors:
        t_r = torch.nn.functional.interpolate(
            t.unsqueeze(0).float(), size=(yolo_imgsz, yolo_imgsz),
            mode='bilinear', align_corners=False
        ).squeeze(0).div_(255.0)
        yolo_tensors.append(t_r)
    yolo_batch = torch.stack(yolo_tensors)
    all_yolo_results = yolo_model(yolo_batch, verbose=False)
    t_yolo = time.time() - t1

    # 3. Маскирование + crop (всё на GPU, без PIL)
    t1 = time.time()
    crop_tensors = []
    for img_tensor, yres, s in zip(img_tensors, all_yolo_results, specs):
        crop_t, _ = apply_masking_gpu(img_tensor, (s["x"], s["y"], s["w"], s["h"]), yres, return_tensor=True)
        crop_tensors.append(crop_t)  # (3, h, w) float32, CUDA
    t_mask = time.time() - t1

    del img_tensors

    # 4. V2 forward: препроцессинг + embed_tensor на GPU (без CPU round-trip)
    t1 = time.time()
    model_size = getattr(model, 'size', 320)
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda', dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda', dtype=torch.float32).view(1, 3, 1, 1)
    
    # Стек всех crop-ов (разные размеры) → resize → стек → нормализация
    # (crop_tensors из apply_masking_gpu return_tensor=True имеют значения 0-255)
    resized = torch.stack([
        torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(model_size, model_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        for t in crop_tensors
    ])  # (N, 3, 320, 320) float32, CUDA
    pv = (resized / 255.0 - mean) / std  # нормализация ImageNet
    
    e = model.embed_tensor(pv)  # (N, 2048)
    import numpy as np
    embs_np = e.cpu().numpy().astype(np.float32)
    embs = embs_np.tolist()
    t_v2 = time.time() - t1

    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"[inference] batch={len(specs)}: {elapsed_ms}ms total"
          f" (yolo={t_yolo*1000:.0f}ms mask={t_mask*1000:.0f}ms v2={t_v2*1000:.0f}ms)"
          f" {len(specs)/max(elapsed_ms,1)*1000:.0f} FPS", flush=True)
    return {"embeddings": embs, "elapsed_ms": elapsed_ms, "n": len(embs)}


@app.post("/internal/attention")
async def attention(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
):
    """Внимание модели к токенам изображения.
    Возвращает эмбеддинг + веса внимания последнего слоя pooling.
    """
    if model is None:
        raise HTTPException(503, f"Модель не загружена: {model_error or 'неизвестная ошибка'}")
    if not hasattr(model, 'embed_with_attention'):
        raise HTTPException(501, "Attention maps не поддерживаются для данной архитектуры модели")

    import numpy as np

    t0 = time.time()
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    crop = img.crop((x, y, x + w, y + h))
    emb, attn = model.embed_with_attention([crop], batch=1)
    elapsed_ms = int((time.time() - t0) * 1000)

    # Первый токен = CLS, 4 регистра, остальные — patch-токены (grid × grid)
    grid = model.size // 16
    n_patches = grid * grid
    patch_attn = attn[0, 5:5 + n_patches].tolist()

    return {
        "embedding": emb[0].tolist(),
        "attention_weights": patch_attn,
        "grid_size": grid,
        "n_tokens": len(attn[0]),
        "elapsed_ms": elapsed_ms,
    }


@app.post("/internal/compare")
async def compare_with_relevance(
    query_image: UploadFile = File(...),
    candidate_image: UploadFile = File(...),
):
    """Grad-ATTN: сравнение двух изображений с визуализацией релевантности.

    Принимает два кропа (query и candidate), вычисляет их эмбеддинги,
    косинусное сходство, и назадпроецирует градиент к patch-токенам.
    Возвращает карты релевантности как data URL изображений-оверлеев.
    """
    if model is None:
        raise HTTPException(503, f"Модель не загружена: {model_error or 'неизвестная ошибка'}")
    if not hasattr(model, 'compare_with_relevance'):
        raise HTTPException(501, "Grad-ATTN не поддерживается для данной архитектуры модели")

    import base64
    import numpy as np

    t0 = time.time()

    q_data = await query_image.read()
    c_data = await candidate_image.read()
    q_img = Image.open(io.BytesIO(q_data)).convert("RGB")
    c_img = Image.open(io.BytesIO(c_data)).convert("RGB")

    q_emb, c_emb, q_rel, c_rel, sim = model.compare_with_relevance(q_img, c_img)

    # Генерация heatmap overlay — цветная колор map, как в cls_attention
    def make_overlay(pil_img, rel_map):
        """Создаёт цветной heatmap overlay из relevance map."""
        w_img, h_img = pil_img.size
        attn_big = np.array(Image.fromarray(rel_map).resize((w_img, h_img), Image.BICUBIC))
        img_np = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
        # Colormap: синий→голубой→зелёный→жёлтый→красный (high=красный, low=синий)
        r = np.interp(attn_big, [0, 0.3, 0.7, 1.0], [0.1, 0.2, 0.9, 1.0])
        g = np.interp(attn_big, [0, 0.3, 0.7, 1.0], [0.3, 0.8, 0.9, 0.1])
        b = np.interp(attn_big, [0, 0.3, 0.7, 1.0], [0.9, 0.6, 0.1, 0.0])
        colormap = np.stack([r, g, b], axis=2)
        blend = 0.6
        result = img_np * (1 - blend) + colormap * blend
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        overlay = Image.fromarray(result)
        buf = io.BytesIO()
        overlay.save(buf, "JPEG", quality=92)
        buf.seek(0)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    q_overlay = make_overlay(q_img, q_rel)
    c_overlay = make_overlay(c_img, c_rel)

    elapsed_ms = int((time.time() - t0) * 1000)

    # Regions: разбиваем карту на 4 квадранта для каждого изображения
    def quad_shares(rel):
        h = rel.shape[0] // 2
        w = rel.shape[1] // 2
        total = rel.sum()
        if total == 0:
            return [0, 0, 0, 0]
        return [
            float(rel[:h, :w].sum() / total),
            float(rel[:h, w:].sum() / total),
            float(rel[h:, :w].sum() / total),
            float(rel[h:, w:].sum() / total),
        ]

    q_shares = quad_shares(q_rel)
    c_shares = quad_shares(c_rel)

    return {
        "query_overlay_url": q_overlay,
        "candidate_overlay_url": c_overlay,
        "similarity": float(sim),
        "elapsed_ms": elapsed_ms,
        "regions": [
            {"name": "Запрос · верх-лево", "share": q_shares[0]},
            {"name": "Запрос · верх-право", "share": q_shares[1]},
            {"name": "Запрос · низ-лево", "share": q_shares[2]},
            {"name": "Запрос · низ-право", "share": q_shares[3]},
            {"name": "Кандидат · верх-лево", "share": c_shares[0]},
            {"name": "Кандидат · верх-право", "share": c_shares[1]},
            {"name": "Кандидат · низ-лево", "share": c_shares[2]},
            {"name": "Кандидат · низ-право", "share": c_shares[3]},
        ],
    }


@app.post("/internal/cls-attention")
async def cls_attention(
    image: UploadFile = File(...),
):
    """CLS self-attention из последнего слоя DINOv3.
    Показывает, на какие области изображения смотрит CLS токен модели.
    """
    if model is None:
        raise HTTPException(503, f"Модель не загружена: {model_error or 'неизвестная ошибка'}")
    if not hasattr(model, 'cls_attention_map'):
        raise HTTPException(501, "CLS attention не поддерживается для данной архитектуры")

    import base64
    import numpy as np

    t0 = time.time()
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    attn_map = model.cls_attention_map(img)

    # Генерация heatmap overlay
    def make_overlay(pil_img, attn):
        """Сплошной overlay: каждый пиксель смесь исходного изображения и heatmap.
        Низкое attention → холодный оттенок (синеватый), высокое → тёплый (красный/жёлтый).
        Никакой прозрачности — эффект виден на всём изображении.
        """
        attn = (attn - attn.min()) / max(attn.max() - attn.min(), 1e-8)
        w_img, h_img = pil_img.size
        # NEAREST чтобы сохранить патчи
        attn_big = np.array(Image.fromarray(attn).resize((w_img, h_img), Image.NEAREST))

        img_np = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0

        # Colormap: синий→голубой→зелёный→жёлтый→красный (high=красный, low=синий)
        # a=0: (0.1, 0.3, 0.9) синий
        # a=0.5: (0.9, 0.9, 0.1) жёлтый
        # a=1.0: (1.0, 0.1, 0.0) красный
        r = np.interp(attn_big, [0, 0.3, 0.7, 1.0], [0.1, 0.2, 0.9, 1.0])
        g = np.interp(attn_big, [0, 0.3, 0.7, 1.0], [0.3, 0.8, 0.9, 0.1])
        b = np.interp(attn_big, [0, 0.3, 0.7, 1.0], [0.9, 0.6, 0.1, 0.0])
        colormap = np.stack([r, g, b], axis=2)

        # Сила наложения: 60% colormap + 40% исходное изображение
        blend = 0.6
        result = img_np * (1 - blend) + colormap * blend
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        overlay = Image.fromarray(result)
        buf = io.BytesIO()
        overlay.save(buf, "JPEG", quality=92)
        buf.seek(0)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    overlay_url = make_overlay(img, attn_map)

    # Regions: 4 квадранта
    h = attn_map.shape[0] // 2
    w = attn_map.shape[1] // 2
    total = attn_map.sum()
    shares = [
        float(attn_map[:h, :w].sum() / total),
        float(attn_map[:h, w:].sum() / total),
        float(attn_map[h:, :w].sum() / total),
        float(attn_map[h:, w:].sum() / total),
    ] if total > 0 else [0.25, 0.25, 0.25, 0.25]

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "overlay_url": overlay_url,
        "elapsed_ms": elapsed_ms,
        "regions": [
            {"name": "верх-лево", "share": shares[0]},
            {"name": "верх-право", "share": shares[1]},
            {"name": "низ-лево", "share": shares[2]},
            {"name": "низ-право", "share": shares[3]},
        ],
    }




