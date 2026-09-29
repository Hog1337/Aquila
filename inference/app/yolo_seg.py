"""
YOLO11s-seg пайплайн: маскирование чужих машин + crop по GT bbox.
Всё на GPU (кроме финальной конверсии в PIL для V2).

Оптимизированная версия: маскирование в 640×640 (без интерполяции до 1080p).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def _compute_iou_tensor(mask: torch.Tensor, x1: int, y1: int, x2: int, y2: int) -> torch.Tensor:
    """IoU между бинарной маской (H,W) и прямоугольником, на GPU."""
    H, W = mask.shape
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(W, x2)
    y2c = min(H, y2)
    if x2c <= x1c or y2c <= y1c:
        return torch.tensor(0.0, device=mask.device)
    gt = torch.zeros((H, W), dtype=mask.dtype, device=mask.device)
    gt[y1c:y2c, x1c:x2c] = 1.0
    inter = (mask & gt).sum()
    union = (mask | gt).sum()
    return inter / union if union > 0 else torch.tensor(0.0, device=mask.device)


def _iou_640(mask: torch.Tensor, gx: int, gy: int, gw: int, gh: int) -> torch.Tensor:
    """IoU между бинарной маской (640,640) и прямоугольником (в 640×640 координатах)."""
    gt = torch.zeros((640, 640), dtype=torch.bool, device=mask.device)
    gt[gy:gy+gh, gx:gx+gw] = True
    inter = (mask & gt).sum()
    union = (mask | gt).sum()
    return inter / union if union > 0 else torch.tensor(0.0, device=mask.device)


def apply_masking_gpu(
    full_tensor: torch.Tensor,
    gt_bbox: tuple[int, int, int, int],
    yolo_results,
    return_tensor: bool = False,
) -> tuple[Image.Image | torch.Tensor, float]:
    """Маскирование чужих машин полностью на GPU (original version).

    Args:
        full_tensor: (3, H, W) uint8 на CUDA.
        gt_bbox: (x, y, w, h).
        yolo_results: результат YOLO.
        return_tensor: если True, возвращает GPU тензор вместо PIL.

    Returns:
        (masked_crop PIL или GPU тензор (3, h, w) float32, iou_best).
    """
    device = full_tensor.device
    _, Horig, Worig = full_tensor.shape
    gx, gy, gw, gh = gt_bbox

    if yolo_results is None or yolo_results.masks is None or len(yolo_results) == 0:
        crop_t = full_tensor[:, gy:gy+gh, gx:gx+gw]
        if return_tensor:
            return crop_t.float(), 0.0
        return Image.fromarray(crop_t.permute(1, 2, 0).byte().cpu().numpy()), 0.0

    masks = yolo_results.masks.data  # (N, 640, 640) float32
    cls_ids = yolo_results.boxes.cls if yolo_results.boxes is not None else None

    # Интерполяция до исходного размера на GPU
    masks_up = F.interpolate(masks.float().unsqueeze(0), size=(Horig, Worig),
                             mode="bilinear", align_corners=False).squeeze(0)

    # Выбираем target (макс. IoU с GT bbox)
    best_iou = 0.0
    best_idx = -1
    for i in range(masks_up.shape[0]):
        if cls_ids is not None and int(cls_ids[i]) != 2:
            continue
        m_bin = (masks_up[i] > 0.5).to(torch.uint8)
        iou = _compute_iou_tensor(m_bin, gx, gy, gx + gw, gy + gh)
        if iou > best_iou:
            best_iou = iou.item()
            best_idx = i

    if best_idx < 0 or best_iou < 0.3:
        crop_t = full_tensor[:, gy:gy+gh, gx:gx+gw]
        if return_tensor:
            return crop_t.float(), 0.0
        return Image.fromarray(crop_t.permute(1, 2, 0).byte().cpu().numpy()), 0.0

    # keep_mask на GPU
    keep = torch.ones((Horig, Worig), dtype=torch.float32, device=device)

    for i in range(masks_up.shape[0]):
        if i == best_idx:
            continue
        if cls_ids is not None and int(cls_ids[i]) != 2:
            continue
        m_bin = (masks_up[i] > 0.5).to(torch.float32)
        overlap = m_bin[gy:gy+gh, gx:gx+gw].sum()
        if overlap > 10:
            keep[m_bin > 0] = 0.0

    keep[gy:gy+gh, gx:gx+gw] = 1.0

    masked = full_tensor.float() * keep.unsqueeze(0)
    crop_t = masked[:, gy:gy+gh, gx:gx+gw]

    if return_tensor:
        return crop_t, best_iou
    crop_pil = Image.fromarray(crop_t.permute(1, 2, 0).byte().cpu().numpy())
    return crop_pil, best_iou


def apply_masking_gpu_fast(
    full_tensor: torch.Tensor,
    resized_640: torch.Tensor,
    gt_bbox: tuple[int, int, int, int],
    orig_size: tuple[int, int],
    yolo_results,
    model_size: int = 320,
) -> tuple[torch.Tensor, float]:
    """Маскирование в 640×640 — без интерполяции до 1080p.

    Args:
        full_tensor: (3, H, W) uint8 CUDA (оригинал, нужен для fallback).
        resized_640: (3, 640, 640) float32, уже resized+normalized для YOLO (0-1).
        gt_bbox: (x, y, w, h) в оригинальном разрешении.
        orig_size: (H, W) оригинального кадра.
        yolo_results: результат YOLO.
        model_size: размер для V2 forward (320).

    Returns:
        (crop_tensor (3, model_size, model_size) float32 CUDA, iou_best).
    """
    device = full_tensor.device
    Horig, Worig = orig_size
    gx, gy, gw, gh = gt_bbox

    if yolo_results is None or yolo_results.masks is None or len(yolo_results) == 0:
        # fallback: crop из оригинала + resize
        crop_t = full_tensor[:, gy:gy+gh, gx:gx+gw].float()
        crop_t = F.interpolate(crop_t.unsqueeze(0), size=(model_size, model_size),
                               mode='bilinear', align_corners=False).squeeze(0)
        return crop_t, 0.0

    masks = yolo_results.masks.data  # (N, 640, 640) float32
    cls_ids = yolo_results.boxes.cls if yolo_results.boxes is not None else None

    # Scale GT bbox → 640×640
    sx, sy = 640.0 / Worig, 640.0 / Horig
    gx_s = max(0, int(gx * sx))
    gy_s = max(0, int(gy * sy))
    gw_s = max(1, int(gw * sx))
    gh_s = max(1, int(gh * sy))
    # clamp
    if gx_s + gw_s > 640: gw_s = 640 - gx_s
    if gy_s + gh_s > 640: gh_s = 640 - gy_s

    # Найти best_idx в 640×640 (без интерполяции!)
    best_iou = 0.0
    best_idx = -1
    for i in range(masks.shape[0]):
        if cls_ids is not None and int(cls_ids[i]) != 2:
            continue
        m_bin = masks[i] > 0.5  # bool mask
        iou = _iou_640(m_bin, gx_s, gy_s, gw_s, gh_s)
        iou_val = iou.item()
        if iou_val > best_iou:
            best_iou = iou_val
            best_idx = i

    if best_idx < 0 or best_iou < 0.3:
        # fallback
        crop_t = full_tensor[:, gy:gy+gh, gx:gx+gw].float()
        crop_t = F.interpolate(crop_t.unsqueeze(0), size=(model_size, model_size),
                               mode='bilinear', align_corners=False).squeeze(0)
        return crop_t, 0.0

    # Создать keep mask в 640×640
    keep = torch.ones((640, 640), dtype=torch.float32, device=device)
    for i in range(masks.shape[0]):
        if i == best_idx:
            continue
        if cls_ids is not None and int(cls_ids[i]) != 2:
            continue
        m_bin = (masks[i] > 0.5).to(torch.float32)
        # Проверяем пересечение с GT bbox (в 640×640)
        overlap = m_bin[gy_s:gy_s+gh_s, gx_s:gx_s+gw_s].sum()
        if overlap > 5:  # порог в 640×640 координатах
            keep[m_bin > 0] = 0.0

    # Force-keep GT bbox
    keep[gy_s:gy_s+gh_s, gx_s:gx_s+gw_s] = 1.0

    # Применяем маску к resized_640
    masked = resized_640 * keep.unsqueeze(0)  # (3, 640, 640)
    crop_640 = masked[:, gy_s:gy_s+gh_s, gx_s:gx_s+gw_s]  # (3, h_s, w_s)

    # Сразу resize до model_size (избавляет от цикла в V2 preproc)
    crop_out = F.interpolate(crop_640.unsqueeze(0), size=(model_size, model_size),
                             mode='bilinear', align_corners=False).squeeze(0)

    return crop_out, best_iou
