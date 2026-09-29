#!/usr/bin/env python3
"""ReID embedding backends: AttnPoolHead (FALCON_FT) и HeadBNNeck.
Автоопределение формата чекпоинта по размеру proj.weight.

FalconFT         — FALCON_FT_best.pt:         AttnPoolHead, 1 query, last layer, 512d
HeadBNNeck V1    — JDNF_BNNECK_PROTO*.pt:     6 queries, layers [15,17,19,21,22,23], LayerNorm, 512d
HeadBNNeck V2    — JDNFV_SUPCON_PROTO_R32*.pt: 6 queries, layers [0,4,8,12,16,20], no LN, 2048d
"""
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
BNNECK_N_LAYERS = 6

# =========================================================================
# LoRA
# =========================================================================
class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=32.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        # Используем dtype base (может быть half (bf16))
        dt = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, in_f, dtype=dt))
        self.lora_B = nn.Parameter(torch.empty(out_f, r, dtype=dt))
        self.scaling = alpha / r
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base(x) + ((x @ self.lora_A.T) @ self.lora_B.T) * self.scaling


def wrap_lora(model, targets, r=16, alpha=32.0):
    """Оборачивает nn.Linear, чьи имена содержат targets. Возвращает число модулей."""
    n = 0
    def rec(parent, prefix):
        nonlocal n
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in full for t in targets):
                setattr(parent, name, LoRALinear(child, r, alpha))
                n += 1
            else:
                rec(child, full)
    rec(model, "")
    return n


# =========================================================================
# AttnPoolHead — для FALCON_FT
# =========================================================================
class AttnPoolHead(nn.Module):
    def __init__(self, tok_dim=1024, proj_dim=512, n_cls=0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(tok_dim))
        self.proj = nn.Linear(tok_dim, proj_dim)
        if n_cls:
            self.cls = nn.Linear(proj_dim, n_cls)

    def embed(self, tokens):
        t = F.normalize(tokens, dim=-1)
        q = F.normalize(self.query, dim=-1)
        w = torch.softmax(t @ q / (t.shape[-1] ** 0.5), dim=1)
        pooled = (t * w.unsqueeze(-1)).sum(1)
        return F.normalize(self.proj(pooled), dim=-1)


# =========================================================================
# HeadBNNeck — для JDNF_* и JDNFV_*
# =========================================================================
class HeadBNNeck(nn.Module):
    """Multi-layer attention-pooling head с LayerNorm + BatchNorm."""
    def __init__(self, tok_dim=1024, proj_dim=512, n_layers=BNNECK_N_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.queries = nn.Parameter(torch.randn(n_layers, tok_dim) * 0.02)
        self.norms = nn.ModuleList([nn.LayerNorm(tok_dim) for _ in range(n_layers)])
        self.proj = nn.Linear(tok_dim * n_layers, proj_dim)
        self.bn = nn.BatchNorm1d(proj_dim)
        self._last_attn_weights = None
        self._keep_graph = False

    def embed(self, tokens_list):
        pooled = []
        for i, tokens in enumerate(tokens_list):
            t = self.norms[i](tokens)
            t = F.normalize(t, dim=-1)
            q = F.normalize(self.queries[i], dim=-1)
            w = torch.softmax(t @ q / (t.shape[-1] ** 0.5), dim=1)
            pooled.append((t * w.unsqueeze(-1)).sum(1))
            if i == len(tokens_list) - 1:
                self._last_attn_weights = w if self._keep_graph else w.detach()
        fused = F.normalize(self.proj(torch.cat(pooled, dim=-1)), dim=-1)
        return fused


# =========================================================================
# Hook helper
# =========================================================================
def _hook_tokens(module, inp, out, store, idx):
    if isinstance(out, tuple):
        out = out[0]
    store[idx] = out


# =========================================================================
# FalconFT — оригинальная модель (AttnPoolHead)
# =========================================================================
class FalconFT:
    """Модель с AttnPoolHead: DINOv3 + LoRA + blocks 16-23 + AttnPool head.
    Чекпоинт: FALCON_FT_best.pt"""
    def __init__(self, dino_dir, ckpt_path, device="auto", size=256):
        from transformers import AutoImageProcessor, AutoModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.size = int(size)
        # Half precision: bf16 если GPU поддерживает, иначе fp16
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            self.half_dtype = torch.bfloat16
        elif device.startswith("cuda"):
            self.half_dtype = torch.float16
        else:
            self.half_dtype = None

        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        self.n_cls = int(ck.get("n_cls", ck["head"]["cls.weight"].shape[0]))

        # Resolve model dir (try -bf16 fallback if original doesn't exist)
        _dino = Path(str(dino_dir))
        if not _dino.exists():
            _bf16 = _dino.with_name(_dino.name + "-bf16")
            if _bf16.exists():
                _dino = _bf16
        self.proc = AutoImageProcessor.from_pretrained(str(_dino))
        self.model = AutoModel.from_pretrained(str(_dino))
        self.model.requires_grad_(False)

        n = wrap_lora(self.model, LORA_TARGETS)
        if n != 96:
            raise RuntimeError(f"LoRA: обёрнуто {n} модулей, ожидается 96")
        loaded = 0
        for name, mod in self.model.named_modules():
            if isinstance(mod, LoRALinear):
                a = ck["lora"].get(name + ".lora_A")
                b = ck["lora"].get(name + ".lora_B")
                if a is None or b is None:
                    raise RuntimeError(f"в чекпоинте нет LoRA-ключа для {name}")
                with torch.no_grad():
                    mod.lora_A.copy_(a)
                    mod.lora_B.copy_(b)
                loaded += 1
        if loaded != 96:
            raise RuntimeError(f"LoRA загружено {loaded}/96")

        self.head = AttnPoolHead(1024, 512, self.n_cls)
        self.head.query.data.copy_(ck["head"]["query"])
        self.head.proj.load_state_dict({
            "weight": ck["head"]["proj.weight"],
            "bias": ck["head"]["proj.bias"],
        })
        if "cls.weight" in ck["head"]:
            self.head.cls.load_state_dict({
                "weight": ck["head"]["cls.weight"],
                "bias": ck["head"]["cls.bias"],
            })

        bl = ck.get("blocks", {})
        if set(bl.keys()) != {"16", "17", "18", "19", "20", "21", "22", "23"}:
            raise RuntimeError(f"blocks в чекпоинте: {sorted(bl.keys())}, ожидается 16-23")
        for li_s, st in bl.items():
            self.model.model.layer[int(li_s)].load_state_dict(st, strict=True)

        self.model.eval()
        self.head.eval()
        self.model.to(self.device)
        self.head.to(self.device)

    @torch.inference_mode()
    def embed_pils(self, pils, batch=32):
        outs = []
        for i in range(0, len(pils), batch):
            chunk = pils[i:i + batch]
            pv = self.proc(images=chunk, size=(self.size, self.size),
                           return_tensors="pt")["pixel_values"].to(self.device)
            if self.half_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self.half_dtype):
                    last = self.model(pixel_values=pv).last_hidden_state
                    e = self.head.embed(last)
            else:
                last = self.model(pixel_values=pv).last_hidden_state
                e = self.head.embed(last)
            outs.append(e.float().cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)


# =========================================================================
# FalconFT_BNNeck — универсальная модель (V1/V2)
# =========================================================================
class FalconFT_BNNeck:
    """Модель с HeadBNNeck. Автоопределение V1 (PROTO) / V2 (R32).

    V1 — JDNF_BNNECK_PROTO*:    512d, r16, α32, слои [15,17,19,21,22,23], LayerNorm
    V2 — JDNFV_SUPCON_PROTO_R32: 2048d, r32, α64, слои [0,4,8,12,16,20], без LayerNorm

    Поддерживает half-precision: bf16 (Ampere+) или fp16 (Turing).
    bf16 численно стабилен для DINOv3, fp16 работает на RTX 2060 (стенд жюри).
    Для ускорения можно использовать единый чекпоинт (*_bf16_full.pt) —
    все веса в одном файле, загрузка одной torch.load.
    """
    def __init__(self, dino_dir, ckpt_path, device="auto", size=320, use_half=True):
        from transformers import AutoImageProcessor, AutoModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.size = int(size)

        # Определяем тип половинной точности:
        #   bf16 — Ampere (cc>=8.0) и новее, стабилен для DINOv3
        #   fp16 — Turing (cc>=7.0), работает на RTX 2060 (стенд жюри)
        #   fp32 — всё остальное (CPU, старые GPU)
        self.use_half = False
        self.half_dtype: torch.dtype | None = None
        if use_half and device.startswith("cuda"):
            if torch.cuda.is_bf16_supported():
                self.half_dtype = torch.bfloat16
                self.use_half = True
                print(f"[reid_embed] using bf16 (cc={torch.cuda.get_device_capability()})", flush=True)
            else:
                # Turing (GTX 16xx, RTX 20xx) — только fp16
                self.half_dtype = torch.float16
                self.use_half = True
                print(f"[reid_embed] bf16 not supported (cc={torch.cuda.get_device_capability()}), using fp16", flush=True)

        if not isinstance(ckpt_path, str):
            ckpt_path = str(ckpt_path)

        # Определяем, откуда загружать DINOv3: при half-точности — *-bf16/ (там safetensors),
        # иначе используем исходный путь (fp32).
        _dino = Path(str(dino_dir))
        if self.use_half:
            bf16_candidate = _dino.with_name(_dino.name + "-bf16")
            if bf16_candidate.exists():
                self._dino_dir = str(bf16_candidate)
            else:
                self._dino_dir = str(_dino)
        else:
            if _dino.exists():
                self._dino_dir = str(_dino)
            else:
                bf16_candidate = _dino.with_name(_dino.name + "-bf16")
                if bf16_candidate.exists():
                    self._dino_dir = str(bf16_candidate)
                else:
                    raise FileNotFoundError(
                        f"DINOv3 directory not found: {_dino} (also tried {_dino.name}-bf16/)"
                    )

        # --- Попытка загрузить единый чекпоинт (backbone + lora + head + blocks) ---
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        if "backbone" in ck:
            # Единый чекпоинт (*_bf16_full.pt): всё уже внутри
            print(f"[reid_embed] loading combined checkpoint: {Path(ckpt_path).name}", flush=True)
            hd = ck["head"]
            proj_dim = hd["proj.weight"].shape[0]
            self.proj_dim = proj_dim

            # Определяем версию
            if proj_dim == 2048:
                lora_r, lora_alpha = 32, 64.0
                expected_blocks = {"14", "15", "16", "17", "18", "19", "20", "21", "22", "23"}
                self._hook_layers = [0, 4, 8, 12, 16, 20]
                self._use_norm = False
                version = "V2 (joint_triple.py, +VeRI-Wild)"
            elif proj_dim == 512:
                lora_r, lora_alpha = 16, 32.0
                expected_blocks = {"16", "17", "18", "19", "20", "21", "22", "23"}
                self._hook_layers = [15, 17, 19, 21, 22, 23]
                self._use_norm = True
                version = "V1 (joint_bnneck_proto.py)"
            else:
                raise RuntimeError(f"Неизвестная размерность proj.weight: {proj_dim}")

            print(f"[reid_embed] detected {version}: dim={proj_dim}, r={lora_r}, α={lora_alpha}", flush=True)

            # Загружаем backbone из сохранённых весов (уже в нужном dtype)
            self.proc = AutoImageProcessor.from_pretrained(self._dino_dir)
            bb_sd = ck["backbone"]
            # Загружаем backbone в нужном dtype (half или fp32)
            bb_dtype = self.half_dtype if self.use_half else torch.float32
            self.model = AutoModel.from_pretrained(
                self._dino_dir, dtype=bb_dtype
            )
            # Загружаем сохранённые веса (скастуются в нужный dtype)
            miss, unex = self.model.load_state_dict(bb_sd, strict=False)
            if miss:
                print(f"[reid_embed] backbone missing keys: {miss[:5]}...", flush=True)
            self.model.requires_grad_(False)

            # LoRA
            n_wrapped = wrap_lora(self.model, LORA_TARGETS, r=lora_r, alpha=lora_alpha)
            n_expected = 24 * len(LORA_TARGETS)  # 96
            if n_wrapped != n_expected:
                raise RuntimeError(f"LoRA: обёрнуто {n_wrapped} модулей, ожидается {n_expected}")

            loaded = 0
            for name, mod in self.model.named_modules():
                if isinstance(mod, LoRALinear):
                    a = ck["lora"].get(name + ".lora_A")
                    b = ck["lora"].get(name + ".lora_B")
                    if a is None or b is None:
                        alt = name.replace("model.", "", 1) if name.startswith("model.") else f"model.{name}"
                        a = ck["lora"].get(alt + ".lora_A") or ck["lora"].get(name + ".lora_A")
                        b = ck["lora"].get(alt + ".lora_B") or ck["lora"].get(name + ".lora_B")
                    if a is None or b is None:
                        raise RuntimeError(f"LoRA-ключ не найден для {name}")
                    with torch.no_grad():
                        mod.lora_A.copy_(a)
                        mod.lora_B.copy_(b)
                    loaded += 1
            if loaded != n_expected:
                raise RuntimeError(f"LoRA загружено {loaded}/{n_expected}")

            # Head
            self.head = HeadBNNeck(1024, proj_dim, BNNECK_N_LAYERS)
            self.head.queries.data.copy_(hd["queries"])
            for i in range(BNNECK_N_LAYERS):
                self.head.norms[i].weight.data.copy_(hd[f"norms.{i}.weight"])
                self.head.norms[i].bias.data.copy_(hd[f"norms.{i}.bias"])
            self.head.proj.load_state_dict({"weight": hd["proj.weight"], "bias": hd["proj.bias"]})
            self.head.bn.load_state_dict({
                "weight": hd["bn.weight"], "bias": hd["bn.bias"],
                "running_mean": hd["bn.running_mean"], "running_var": hd["bn.running_var"],
                "num_batches_tracked": hd["bn.num_batches_tracked"],
            })

            # Blocks
            bl = ck.get("blocks", {})
            loaded_blocks = set(bl.keys())
            if loaded_blocks != expected_blocks:
                warnings.warn(
                    f"blocks в чекпоинте: {sorted(loaded_blocks)}, "
                    f"ожидалось {sorted(expected_blocks)}. Загружаем что есть."
                )
            for li_s, st in bl.items():
                self.model.model.layer[int(li_s)].load_state_dict(st, strict=True)

        else:
            # --- Раздельный чекпоинт (только LoRA + head + blocks) ---
            hd = ck["head"]
            proj_dim = hd["proj.weight"].shape[0]

            # Определяем версию
            if proj_dim == 2048:
                lora_r, lora_alpha = 32, 64.0
                expected_blocks = {"14", "15", "16", "17", "18", "19", "20", "21", "22", "23"}
                self._hook_layers = [0, 4, 8, 12, 16, 20]
                self._use_norm = False
                version = "V2 (joint_triple.py, +VeRI-Wild)"
            elif proj_dim == 512:
                lora_r, lora_alpha = 16, 32.0
                expected_blocks = {"16", "17", "18", "19", "20", "21", "22", "23"}
                self._hook_layers = [15, 17, 19, 21, 22, 23]
                self._use_norm = True
                version = "V1 (joint_bnneck_proto.py)"
            else:
                raise RuntimeError(f"Неизвестная размерность proj.weight: {proj_dim}")

            self.proj_dim = proj_dim
            print(f"[reid_embed] detected {version}: dim={proj_dim}, r={lora_r}, α={lora_alpha}", flush=True)

            # Загружаем backbone из self._dino_dir
            self.proc = AutoImageProcessor.from_pretrained(self._dino_dir)
            bb_dtype = self.half_dtype if self.use_half else torch.float32
            dtype_label = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}.get(bb_dtype, str(bb_dtype))
            print(f"[reid_embed] loading backbone ({dtype_label}) from {Path(self._dino_dir).name}/", flush=True)
            self.model = AutoModel.from_pretrained(self._dino_dir, dtype=bb_dtype)

            self.model.requires_grad_(False)

            # LoRA
            n_wrapped = wrap_lora(self.model, LORA_TARGETS, r=lora_r, alpha=lora_alpha)
            n_expected = 24 * len(LORA_TARGETS)  # 96
            if n_wrapped != n_expected:
                raise RuntimeError(f"LoRA: обёрнуто {n_wrapped} модулей, ожидается {n_expected}")

            loaded = 0
            for name, mod in self.model.named_modules():
                if isinstance(mod, LoRALinear):
                    a = ck["lora"].get(name + ".lora_A")
                    b = ck["lora"].get(name + ".lora_B")
                    if a is None or b is None:
                        alt = name.replace("model.", "", 1) if name.startswith("model.") else f"model.{name}"
                        a = ck["lora"].get(alt + ".lora_A") or ck["lora"].get(name + ".lora_A")
                        b = ck["lora"].get(alt + ".lora_B") or ck["lora"].get(name + ".lora_B")
                    if a is None or b is None:
                        raise RuntimeError(f"LoRA-ключ не найден для {name}")
                    with torch.no_grad():
                        mod.lora_A.copy_(a)
                        mod.lora_B.copy_(b)
                    loaded += 1
            if loaded != n_expected:
                raise RuntimeError(f"LoRA загружено {loaded}/{n_expected}")

            # Head
            self.head = HeadBNNeck(1024, proj_dim, BNNECK_N_LAYERS)
            self.head.queries.data.copy_(hd["queries"])
            for i in range(BNNECK_N_LAYERS):
                self.head.norms[i].weight.data.copy_(hd[f"norms.{i}.weight"])
                self.head.norms[i].bias.data.copy_(hd[f"norms.{i}.bias"])
            self.head.proj.load_state_dict({"weight": hd["proj.weight"], "bias": hd["proj.bias"]})
            self.head.bn.load_state_dict({
                "weight": hd["bn.weight"], "bias": hd["bn.bias"],
                "running_mean": hd["bn.running_mean"], "running_var": hd["bn.running_var"],
                "num_batches_tracked": hd["bn.num_batches_tracked"],
            })

            # Blocks
            bl = ck.get("blocks", {})
            loaded_blocks = set(bl.keys())
            if loaded_blocks != expected_blocks:
                warnings.warn(
                    f"blocks в чекпоинте: {sorted(loaded_blocks)}, "
                    f"ожидалось {sorted(expected_blocks)}. Загружаем что есть."
                )
            for li_s, st in bl.items():
                self.model.model.layer[int(li_s)].load_state_dict(st, strict=True)

        # Переводим head в half если нужно
        if self.half_dtype == torch.bfloat16:
            self.head = self.head.bfloat16()
        elif self.half_dtype == torch.float16:
            self.head = self.head.half()

        self.model.eval()
        self.head.eval()
        self.model.to(self.device)
        self.head.to(self.device)

        # Опциональная компиляция backbone
        self._compiled = False

    def compile_backbone(self, mode: str = "default"):
        """Compile backbone с torch.compile для ускорения (1.3-1.5x).
        Пропускается на GPU без bf16 (Turing/RTX 2060) — compile не даст ускорения."""
        if self._compiled:
            return
        if self.half_dtype != torch.bfloat16:
            print(f"[reid_embed] skipping compile on non-bf16 GPU ({self.half_dtype})", flush=True)
            return
        try:
            self.model = torch.compile(self.model, mode=mode)
            self._compiled = True
            print(f"[reid_embed] backbone compiled (mode={mode})", flush=True)
        except Exception as e:
            print(f"[reid_embed] torch.compile failed: {e}", flush=True)

    def _process_tokens(self):
        """Собирает токены из hook-output, применяя LayerNorm если нужно."""
        ordered = sorted(self._tokens.keys())
        if self._use_norm:
            return [self.model.norm(self._tokens[k]) for k in ordered]
        else:
            return [self._tokens[k] for k in ordered]

    @torch.inference_mode()
    def embed_tensor(self, pv: torch.Tensor) -> torch.Tensor:
        """Прямой forward из тензора pixel_values (N, 3, H, W) на self.device.
        Использует output_hidden_states=True вместо forward hooks
        для совместимости с torch.compile.

        Returns:
            Tensor (N, proj_dim) float32, L2-normalized, на self.device.
        """
        # Приводим вход к тому же dtype что и модель
        if self.use_half:
            pv = pv.to(self.half_dtype)
            with torch.autocast(device_type="cuda", dtype=self.half_dtype):
                out = self.model(pixel_values=pv, output_hidden_states=True)
        else:
            out = self.model(pixel_values=pv, output_hidden_states=True)

        # hidden_states: tuple of (N, L, D) для каждого слоя (включая embedding)
        hs = out.hidden_states
        # Выбираем нужные слои + norm если нужно
        if self._use_norm:
            tokens = [self.model.norm(hs[li + 1]) for li in self._hook_layers]
        else:
            tokens = [hs[li + 1] for li in self._hook_layers]
        e = self.head.embed(tokens)
        return e.float()

    @torch.inference_mode()
    def embed_pils(self, pils, batch=32):
        """Список PIL(RGB) -> np.ndarray (N, proj_dim) float32, L2-norm."""
        outs = []
        for i in range(0, len(pils), batch):
            chunk = pils[i:i + batch]
            pv = self.proc(images=chunk, size=(self.size, self.size),
                           return_tensors="pt")["pixel_values"].to(self.device)
            e = self.embed_tensor(pv)
            outs.append(e.cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    @torch.inference_mode()
    def embed_with_attention(self, pils, batch=1):
        embs = []
        all_attn = []
        for i in range(0, len(pils), batch):
            chunk = pils[i:i + batch]
            pv = self.proc(images=chunk, size=(self.size, self.size),
                           return_tensors="pt")["pixel_values"].to(self.device)
            e = self.embed_tensor(pv)
            embs.append(e.cpu().numpy())
            all_attn.append(self.head._last_attn_weights.float().cpu().numpy())
        return (np.concatenate(embs, axis=0).astype(np.float32),
                np.concatenate(all_attn, axis=0).astype(np.float32))

    @torch.inference_mode()
    def compare_with_relevance(self, query_pil, candidate_pil):
        """Сравнивает два изображения, возвращает карты внимания pooling head.
        Использует веса внимания последнего query вектора HeadBNNeck — они показывают,
        на какие патчи модель опирается при формировании эмбеддинга.
        Возвращает: (q_emb, c_emb, q_rel, c_rel, similarity)."""
        # Получаем эмбеддинги обычным путём (embed_tensor с output_hidden_states)
        pils = [query_pil, candidate_pil]
        embs = self.embed_pils(pils, batch=2)
        q_emb, c_emb = embs[0], embs[1]
        # Косинусное сходство
        sim = float((q_emb / np.linalg.norm(q_emb)) @ (c_emb / np.linalg.norm(c_emb)))

        # Внимание pooling head — _last_attn_weights: вес последнего query (index 5) к токенам последнего слоя
        # shape: (batch, n_tokens) — (2, 405) где 1 CLS + 4 register + 400 patch
        grid = self.size // 16
        attn = self.head._last_attn_weights.float().cpu().numpy()  # (2, 405)

        def get_patch_attn(img_idx):
            patch = attn[img_idx, 5:]  # убираем CLS + register
            return patch.reshape(grid, grid)

        q_rel = get_patch_attn(0)
        c_rel = get_patch_attn(1)

        # Нормализация 0-1 для визуализации
        def _norm(m):
            mn, mx = m.min(), m.max()
            if mx - mn > 1e-8:
                return (m - mn) / (mx - mn)
            return m * 0 + 0.5
        q_rel = _norm(q_rel)
        c_rel = _norm(c_rel)

        return (q_emb.astype(np.float32), c_emb.astype(np.float32),
                q_rel.astype(np.float32), c_rel.astype(np.float32), sim)

    @torch.inference_mode()
    def cls_attention_map(self, pil_img):
        old_attn = self.model.config._attn_implementation
        self.model.config._attn_implementation = "eager"
        pv = self.proc(images=pil_img, size=(self.size, self.size),
                       return_tensors="pt")["pixel_values"].to(self.device)
        if self.use_half:
            pv = pv.to(self.half_dtype)
        out = self.model(pixel_values=pv, output_attentions=True)
        last_attn = out.attentions[-1]
        cls_w = last_attn[0, :, 0, :].mean(dim=0)
        grid = self.size // 16  # patch_size = 16 for ViT-L/16
        # cls_w[0] = CLS, cls_w[1:5] = register tokens, cls_w[5:] = patch tokens
        patch_w = cls_w[5:].cpu().float().numpy()
        self.model.config._attn_implementation = old_attn
        return patch_w.astype(np.float32).reshape(grid, grid)

    def close(self):
        if hasattr(self, '_handles'):
            for h in self._handles:
                h.remove()
            self._handles.clear()


# =========================================================================
# Фабрика: автоопределение типа чекпоинта
# =========================================================================
def load_model(dino_dir, ckpt_path, device="auto", size="auto"):
    """Определяет тип чекпоинта по ключам и загружает соответствующую модель.

    - AttnPoolHead: в head есть ключ 'query' (одномерный) → FalconFT
    - HeadBNNeck V1: proj.weight [512, 6144] → FalconFT_BNNeck (LN, слои 15-23)
    - HeadBNNeck V2: proj.weight [2048, 6144] → FalconFT_BNNeck (без LN, слои 0-20)
    """
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    hd = ck.get("head", {})

    if "queries" in hd and len(hd["queries"].shape) == 2:
        proj_dim = hd["proj.weight"].shape[0]
        if size == "auto":
            size = 320
        print(f"[reid_embed] detected BNNeck head ({proj_dim}d) — loading FalconFT_BNNeck ({size}px)", flush=True)
        model = FalconFT_BNNeck(dino_dir, ckpt_path, device=device, size=size)
    elif "query" in hd and len(hd["query"].shape) == 1:
        if size == "auto":
            size = 256
        print(f"[reid_embed] detected AttnPool head — loading FalconFT ({size}px)", flush=True)
        model = FalconFT(dino_dir, ckpt_path, device=device, size=size)
    else:
        raise RuntimeError(
            f"Не удалось определить тип чекпоинта. "
            f"Ключи head: {list(hd.keys())}."
        )

    test_pil = Image.new("RGB", (size, size), (128, 128, 128))
    test_emb = model.embed_pils([test_pil], batch=1)
    print(f"[reid_embed] test embedding dim={test_emb.shape[1]}, device={model.device}", flush=True)

    return model
