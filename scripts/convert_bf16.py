#!/usr/bin/env python3
"""Convert DINOv3 ViT-L/16 weights to bf16 (Brain Floating Point 16).

bf16 has the same exponent range as fp32 (8 bits) — numerically stable
for DINOv3, unlike fp16 which produces NaN due to LayerNorm/Softmax overflow.

1. Saves original fp32 weights as backup (if not already backed up)
2. Converts to bf16 and saves to dinov3-vitl16-bf16/
3. Also saves the combined bf16 checkpoint for quick loading
"""
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoModel, AutoConfig

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "inference" / "weights"
DINO_DIR = WEIGHTS_DIR / "dinov3-vitl16"
DINO_BF16_DIR = WEIGHTS_DIR / "dinov3-vitl16-bf16"
CKPT_PATH = WEIGHTS_DIR / "JDNFV_MASKED_FT_best_fp16.pt"
OUT_CKPT = WEIGHTS_DIR / "JDNFV_MASKED_FT_bf16_full.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Step 1: Backup fp32 weights ---
    backup_dir = WEIGHTS_DIR / "dinov3-vitl16-fp32-backup"
    if not backup_dir.exists():
        print(f"Backing up fp32 DINOv3 to {backup_dir} ...")
        shutil.copytree(DINO_DIR, backup_dir)
        print("  done.")
    else:
        print(f"Backup already exists at {backup_dir}, skipping.")

    # --- Step 2: Load fp32 DINOv3 and convert to bf16 ---
    if DINO_BF16_DIR.exists():
        print(f"bf16 DINOv3 already exists at {DINO_BF16_DIR}, skipping conversion.")
    else:
        print("Loading fp32 DINOv3 from safetensors ...")
        model = AutoModel.from_pretrained(str(DINO_DIR))
        model = model.to(dtype=torch.bfloat16)  # convert to bf16
        model.eval()

        print(f"Saving bf16 DINOv3 to {DINO_BF16_DIR} ...")
        DINO_BF16_DIR.mkdir(exist_ok=True)

        # Save config files (copy from original) — .gitattributes carries the
        # HF *.safetensors filter=lfs rule; without it this file commits as a
        # plain blob instead of an LFS pointer (see git history of this dir).
        for cfg_file in ["config.json", "configuration.json", "preprocessor_config.json", ".gitattributes"]:
            src = DINO_DIR / cfg_file
            if src.exists():
                shutil.copy2(src, DINO_BF16_DIR / cfg_file)

        # Save model weights as safetensors
        state_dict = {k: v.contiguous().bfloat16() for k, v in model.state_dict().items()}
        safetensors_path = DINO_BF16_DIR / "model.safetensors"
        save_file(state_dict, str(safetensors_path))
        print(f"  saved: {safetensors_path} ({safetensors_path.stat().st_size / 1e9:.2f} GB)")
        del model, state_dict
        torch.cuda.empty_cache() if device == "cuda" else None

    # --- Step 3: Combine into one file ---
    if not OUT_CKPT.exists():
        print(f"\nLoading bf16 DINOv3 backbone ...")
        model = AutoModel.from_pretrained(str(DINO_BF16_DIR))
        model.eval()

        print(f"Loading LoRA+head checkpoint: {CKPT_PATH.name} ...")
        ck = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)

        # Combine: backbone + lora + head + blocks
        combined = {
            "backbone": {k: v.contiguous().bfloat16() for k, v in model.state_dict().items()},
            "lora": {},
            "head": {},
            "blocks": {},
        }
        for k, v in ck["lora"].items():
            combined["lora"][k] = v.contiguous().bfloat16() if isinstance(v, torch.Tensor) else v
        for k, v in ck["head"].items():
            combined["head"][k] = v.contiguous().bfloat16() if isinstance(v, torch.Tensor) else v
        for k, v in ck.get("blocks", {}).items():
            combined["blocks"][k] = {kk: vv.contiguous().bfloat16() for kk, vv in v.items()}

        # Metadata
        combined["proj_dim"] = ck.get("proj_dim", 2048)
        combined["lora_r"] = ck.get("lora_r", 32)
        combined["lora_alpha"] = ck.get("lora_alpha", 64.0)
        combined["hook_layers"] = ck.get("hook_layers", [0, 4, 8, 12, 16, 20])

        print(f"Saving combined bf16 checkpoint: {OUT_CKPT.name} ...")
        torch.save(combined, str(OUT_CKPT))
        print(f"  saved: {OUT_CKPT} ({OUT_CKPT.stat().st_size / 1e9:.2f} GB)")
        del model, ck, combined
        torch.cuda.empty_cache() if device == "cuda" else None
    else:
        print(f"Combined checkpoint already exists, skipping.")

    print("\nDone! Files:")
    print(f"  fp32 backup:   {backup_dir}")
    print(f"  bf16 backbone: {DINO_BF16_DIR}")
    print(f"  combined bf16: {OUT_CKPT}")


if __name__ == "__main__":
    main()
