# -*- coding: utf-8 -*-
"""
InternVL-3.5 视觉特征 -> 训练专用 NPZ (X, y, paths)
- 输入 CSV: 至少包含列 ["image_path", "category"]；其他列可有可无
  * 支持有/无表头
  * 支持路径中含逗号（默认取第1列为路径、第2列为类别；若检测到表头则按列名取）
- 输出 NPZ: { X: (N,D) float32, y: (N,) str, paths: (N,) str }
- 模型: 本地 InternVL-3.5 目录 或 HF 仓库名（默认仅本地缓存，--allow-remote 可放开）
"""

import os, csv, argparse, warnings
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM


# ---------------- CSV 读取 ----------------

def read_csv_flexible(csv_path: str) -> List[Tuple[str, str]]:
    """
    优先按表头读取 'image_path' 与 'category'；
    若无表头或表头缺失，则退回为每行取第1列为路径、第2列为类别。
    允许路径中含逗号（用 csv 库解析）。
    """
    rows: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        # 尝试 DictReader（有表头场景）
        try:
            dr = csv.DictReader(f)
            if dr.fieldnames and ("image_path" in dr.fieldnames or "image" in dr.fieldnames) and "category" in dr.fieldnames:
                for r in dr:
                    p = (r.get("image_path") or r.get("image") or "").strip()
                    lab = (r.get("category") or "").strip()
                    if p:
                        rows.append((p, lab))
                if rows:
                    return rows
        except Exception:
            f.seek(0)

        # 无表头：普通 reader，取第1列为路径、第2列为类别
        f.seek(0)
        rr = csv.reader(f)
        for r in rr:
            if not r:
                continue
            if len(r) == 1:
                rows.append((r[0].strip(), ""))  # 只有路径，类别空
            else:
                rows.append((r[0].strip(), r[1].strip()))
    # 过滤掉可能的标题行“image_path,category,...”
    if rows and rows[0][0].lower() == "image_path":
        rows = rows[1:]
    return rows


def ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------- 视觉塔获取 ----------------

def get_internvl_vision_module(model: torch.nn.Module) -> torch.nn.Module:
    """
    常见字段：model.visual_encoder / model.vision_model / visual_encoder / vision_model
    """
    if hasattr(model, "model"):
        sub = model.model
        for name in ["visual_encoder", "vision_model"]:
            m = getattr(sub, name, None)
            if m is not None:
                return m
    for name in ["visual_encoder", "vision_model"]:
        m = getattr(model, name, None)
        if m is not None:
            return m
    raise RuntimeError("未找到视觉编码器（visual_encoder / vision_model）。")


# ---------------- 特征提取 ----------------

@torch.inference_mode()
def extract_feats(
    model: torch.nn.Module,
    image_processor,
    paths: List[str],
    dtype: torch.dtype = torch.float16,
    batch_size: int = 8,
    take: str = "mean_patch",       # 'mean_patch' | 'cls' | 'mean_projected'
    project: bool = False,          # 是否通过 projector
) -> Tuple[np.ndarray, List[str]]:
    feats, kept_paths = [], []

    vision = get_internvl_vision_module(model)
    try:
        device = next(vision.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 找 projector（可能不存在）
    projector = None
    for name in ["multi_modal_projector", "mm_projector", "visual_projection", "projector"]:
        projector = getattr(model, name, None) or getattr(getattr(model, "model", object()), name, None)
        if projector is not None:
            break

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224), (0, 0, 0)))  # 占位黑图

        pix = image_processor(images=images, return_tensors="pt")
        pixel_values = pix["pixel_values"].to(device=device, dtype=dtype)

        # 4D: (B,C,H,W)；5D: (B,N,C,H,W) -> 展平后再在 N 维均值
        restore = None
        if pixel_values.ndim == 5:
            B, N, C, H, W = pixel_values.shape
            if N == 1:
                pixel_values = pixel_values.squeeze(1)
            else:
                pixel_values = pixel_values.reshape(B * N, C, H, W)
                restore = (B, N)

        # 前向
        vis_out = vision(pixel_values)
        last_hidden = getattr(vis_out, "last_hidden_state", None)
        pooled = getattr(vis_out, "pooler_output", None)
        if last_hidden is None:
            last_hidden = vis_out[0] if isinstance(vis_out, (tuple, list)) else vis_out

        # 还原 & 在 N 维聚合
        if restore is not None:
            B, N = restore
            if last_hidden.ndim == 3:
                last_hidden = last_hidden.reshape(B, N, last_hidden.size(1), last_hidden.size(2)).mean(dim=1)
            elif last_hidden.ndim == 2:
                last_hidden = last_hidden.reshape(B, N, last_hidden.size(1)).mean(dim=1)
            if pooled is not None and pooled.ndim == 2:
                pooled = pooled.reshape(B, N, pooled.size(1)).mean(dim=1)

        # 选特征
        if take == "cls" and pooled is not None:
            emb = pooled
        else:
            emb = last_hidden
            if emb.ndim == 3:  # (B,T,D) → token 均值
                emb = emb.mean(dim=1)

        # 可选 projector
        if projector is not None and (project or take == "mean_projected"):
            try:
                x = emb
                if x.ndim == 2:
                    x = x[:, None, :]
                emb = projector(x).mean(dim=1)
            except Exception as e:
                warnings.warn(f"projector 失败，回退未投影特征：{e}")

        feats.append(emb.float().cpu())
        kept_paths.extend(batch_paths)

    X = torch.cat(feats, dim=0).numpy()
    return X, kept_paths


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(description="InternVL-3.5 视觉特征 -> 训练专用 NPZ (X,y,paths)")
    ap.add_argument("--csv", required=True, help="输入 CSV（需含 image_path, category）")
    ap.add_argument("--model", required=True, help="本地 InternVL-3.5 目录 或 HF 仓库名")
    ap.add_argument("--out_npz", required=True, help="输出 npz 路径（含目录将自动创建）")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--take", default="mean_patch", choices=["mean_patch", "cls", "mean_projected"])
    ap.add_argument("--project", action="store_true", help="通过 projector 再均值（或直接用 --take mean_projected）")
    ap.add_argument("--allow-remote", action="store_true", help="允许从 HF 远程拉取（默认仅本地）")
    args = ap.parse_args()

    pairs = read_csv_flexible(args.csv)
    if not pairs:
        raise RuntimeError(f"CSV 为空或无法解析: {args.csv}")

    # 规范化标签：去首尾空格
    paths = [p for p, _ in pairs]
    labels = [(l or "").strip() for _, l in pairs]
    print(f"[Data] samples={len(paths)}  unique_labels={len(set(labels))}")

    # dtype & 加载
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    local_only = not args.allow_remote
    print(f"[Model] loading from {args.model} (local_files_only={local_only}) ...")
    image_processor = AutoImageProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=local_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_only,
    )

    # 抽特征
    print("[Feat] extracting ...")
    X, kept_paths = extract_feats(
        model=model,
        image_processor=image_processor,
        paths=paths,
        dtype=torch_dtype,
        batch_size=args.batch_size,
        take=args.take,
        project=args.project or (args.take == "mean_projected"),
    )
    print(f"[Feat] done: shape={X.shape} dtype={X.dtype}")

    # 保存
    ensure_dir(os.path.dirname(args.out_npz))
    np.savez(args.out_npz, X=X.astype("float32"), y=np.array(labels), paths=np.array(kept_paths))
    print(f"[Save] npz -> {args.out_npz}  (keys: X, y, paths)")
    print("✅ 完成，可直接用于 train_diagnoser.py")


if __name__ == "__main__":
    """
    运行示例：
    # 训练集
    python exctract_Intervl3.5.py  \
      --csv   /data1/home/pankun/MCP/data/train_gt.csv \
      --model /data1/models/InternVL3_5-8B \
      --out_npz /data1/home/pankun/MCP/feats/internvl_train_diag.npz \
      --dtype float16 --batch_size 4 --take mean_patch

    # 验证集
    python exctract_Intervl3.5.py \
      --csv   /data1/home/pankun/MCP/data/val_gt.csv \
      --model /data1/models/InternVL3_5-8B \
      --out_npz /data1/home/pankun/MCP/feats/internvl_val_diag.npz \
      --dtype float16 --batch_size 4 --take mean_patch
    """
    main()
