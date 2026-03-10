# -*- coding: utf-8 -*-
"""
IllusionFixedPipelines MCP Server
---------------------------------
该文件是“错觉图识别系统”的固定策略服务端（通过 MCP 协议与 client 通信）。
根据图像分类标签（AI_unsafe_illusion / HighLowFreq / DoubleLayer 等），
返回预处理结果图像（outs）及推荐识别提示词（vlm_prompt）。

用法：
    python mcp_server.py        # 被 mcp_client.py 以 stdio 模式自动调用
"""

import os, math, json
import numpy as np
from typing import List, Dict
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from mcp.server.fastmcp import FastMCP
import math
import cv2
import numpy as np

try:
    import cv2
    HAS_CV = True
except Exception:
    HAS_CV = False

mcp = FastMCP("IllusionFixedPipelines")

# =========================================================
# ---------- 工具函数 ----------
# =========================================================

def _ensure_dir(p: str):
    """确保输出目录存在"""
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

def _save(p, arr):
    """保存灰度图像"""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(p)

def _to_gray_arr(path) -> np.ndarray:
    """读图为灰度数组"""
    return np.array(Image.open(path).convert("L"))

def _enhance_contrast_sharp(gray_u8: np.ndarray, sharp=1.4, thr=2):
    """自动对比度增强 + 轻锐化"""
    im = Image.fromarray(gray_u8)
    im = ImageOps.autocontrast(im, cutoff=1)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2,
                                           percent=int(sharp*100),
                                           threshold=thr))
    return np.array(im)

# =========================================================
# ---------- 通用频域陷波函数 ----------
# =========================================================
def _fft_notch(gray: np.ndarray, ring=None, peaks=8, bw=None, k_sigma=3.0, harm=2, viz_path=None):
    """
    更稳的频域陷波（条纹抑制）：
    - 对数幅度谱 + 背景（高斯）减除，提显离散峰
    - 在中心圆环外找峰，阈值 = 中位数 + k_sigma*IQR 标准化
    - 自适应带宽/中心圈半径，按图像短边比例缩放
    - 对每个峰连同对称峰与若干谐波一并凹陷
    - 可选保存频谱可视化（viz_path）
    """
    g = gray.astype(np.float32)
    H, W = g.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]

    # 1) FFT & 对数谱
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F)
    logmag = np.log1p(mag)

    # 2) 自适应 ring/bw
    short = min(H, W)
    if ring is None:
        ring = max(10, int(0.035 * short))     # ~3.5% 半径内视为低频
    if bw is None:
        bw = max(8, int(0.02 * short))         # ~2% 的带宽

    # 3) 背景减除，提显峰
    if HAS_CV:
        bg = cv2.GaussianBlur(logmag.astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=3.0)
        resp = logmag - bg
    else:
        resp = logmag

    # 4) 屏蔽中心圆，计算阈值
    resp = resp.copy()
    resp[(yy - cy) ** 2 + (xx - cx) ** 2 <= ring * ring] = 0.0
    flat = resp[resp > 0]
    if flat.size == 0:
        mask = np.ones_like(mag, np.float32)
        out = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
        out -= out.min()
        out = out / max(out.max(), 1e-6) * 255
        return out.astype(np.uint8)

    med = np.median(flat)
    q1, q3 = np.percentile(flat, [25, 75])
    iqr = max(q3 - q1, 1e-6)
    thr = med + k_sigma * iqr

    cand = np.argwhere(resp > thr)
    if cand.size == 0:
        cand = np.argwhere(resp > med)  # 退而求其次

    # 5) 取最强 peaks 个候选（避免取太多）
    vals = resp[cand[:, 0], cand[:, 1]]
    order = np.argsort(-vals)
    sel = cand[order[:peaks]]

    # 6) 构造掩膜：对称点 + 谐波（同方向乘以 2,3...）
    mask = np.ones_like(mag, np.float32)

    def notch_at(r, c, w):
        rr = (yy - r) ** 2 + (xx - c) ** 2 <= (w * w)
        mask[rr] = 0.0

    for r, c in sel:
        # 距离向量
        vr, vc = r - cy, c - cx
        if vr == 0 and vc == 0:
            continue

        # 主峰 + 对称峰
        notch_at(r, c, bw)
        notch_at(2 * cy - r, 2 * cx - c, bw)

        # 谐波（同方向等比例延伸）
        for h in range(2, harm + 1):
            rr = int(round(cy + h * vr))
            cc = int(round(cx + h * vc))
            rs = int(round(cy - h * vr))
            cs = int(round(cx - h * vc))
            # 边界检查
            if 0 <= rr < H and 0 <= cc < W: notch_at(rr, cc, max(2, int(bw * 0.9)))
            if 0 <= rs < H and 0 <= cs < W: notch_at(rs, cs, max(2, int(bw * 0.9)))

    # 7) 可视化（调参用）
    if viz_path:
        try:
            viz = (resp - resp.min()) / max(resp.max() - resp.min(), 1e-6)
            viz = (viz * 255).astype(np.uint8)
            _save(viz_path, viz)
        except Exception:
            pass

    # 8) 逆变换 & 归一化
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
    out -= out.min()
    if out.max() > 0:
        out = out / out.max() * 255
    return out.astype(np.uint8)


# =========================================================
# ---------- 1. AI_unsafe_illusion：读取低频信息 ----------
# =========================================================
@mcp.tool()
def tp_ai_lowfreq_read(src: str, outdir: str = "", down: float = 0.3, blur: float = 2.5) -> dict:
    """
    策略：缩小 + 轻模糊 → 提取低频区域的可读信息。
    输出：一张低频图。
    """
    _ensure_dir(outdir or os.path.dirname(src))
    base = os.path.join(outdir or os.path.dirname(src),
                        os.path.splitext(os.path.basename(src))[0])

    im = Image.open(src).convert("L")
    w, h = im.size
    im_small = im.resize((max(1, int(w * down)), max(1, int(h * down))), Image.LANCZOS)
    im_small = im_small.filter(ImageFilter.GaussianBlur(radius=blur))
    low = np.array(ImageOps.autocontrast(im_small, cutoff=1))

    p = f"{base}_ai_lowfreq.png"
    _save(p, low)

    # return {
    #     "outs": [os.path.abspath(p)],
    #     "vlm_prompt": (
    #                     "Identify the content in the picture.\n"
    #                     "Strict output rules:\n"
    #                     "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
    #                     "- No explanations, no punctuation, no quotes, no tags.\n"
    #                     "- If any letter is uncertain, replace it with '?'.\n"
    #                     "Now output only the word."
    #     ),
    #     "meta": {"down": down, "blur": blur}
    # }
    return {
        "outs": [os.path.abspath(p)],
        "vlm_prompt": (
            "You are viewing an image that has been down-sampled and blurred to reveal hidden words "
            "embedded in the low-frequency domain. The preprocessing removed fine textures and "
            "emphasized large blurry letter patterns.\n\n"
            "Step 1 – Observe: Look across the image for broad, faint shapes that resemble letters "
            "when viewed from a distance or with relaxed focus.\n"
            "Step 2 – Infer: Combine those shapes into a plausible English word that a human might "
            "perceive under these low-frequency conditions.\n"
            "Step 3 – Verify: Check that the overall structure looks like a real word "
            "(letters evenly spaced, natural pattern).\n"
            "Step 4 – Output: Write only that single English word in lowercase letters (a–z). "
            "Do not include spaces or punctuation. If any character is uncertain, replace it with '?'.\n\n"
            "Respond only with the final word."
        ),
        "meta": {"down": down, "blur": blur}
    }


# =========================================================
# ---------- 2. HighLowFreq：高低频双视图 ----------
# =========================================================
@mcp.tool()
def tp_highlow_dualpass(src: str, outdir: str = "", down: float = 0.5,
                        blur_low: float = 2.5, hp_radius: float = 1.2, hp_amount: int = 160) -> dict:
    """
    策略：输出两张图
      - 第一张：低频（缩小+模糊）
      - 第二张：高频（锐化/USM）
    """
    _ensure_dir(outdir or os.path.dirname(src))
    base = os.path.join(outdir or os.path.dirname(src),
                        os.path.splitext(os.path.basename(src))[0])

    # 低频视图
    im = Image.open(src).convert("L")
    w, h = im.size
    low_small = im.resize((max(1, int(w * down)), max(1, int(h * down))), Image.LANCZOS)
    low_small = low_small.filter(ImageFilter.GaussianBlur(radius=blur_low))
    low = np.array(ImageOps.autocontrast(low_small, cutoff=1))
    p_low = f"{base}_lowfreq.png"
    _save(p_low, low)

    # 高频视图
    hi = Image.open(src).convert("L")
    hi = hi.filter(ImageFilter.UnsharpMask(radius=hp_radius,
                                           percent=hp_amount,
                                           threshold=2))
    hi = ImageOps.autocontrast(hi, cutoff=1)
    p_hi = f"{base}_highfreq.png"
    _save(p_hi, np.array(hi))

    # return {
    #     "outs": [os.path.abspath(p_low), os.path.abspath(p_hi)],
    #     "vlm_prompt": (
    #         "本张图只输出一行小写英文字母的单词。\n"
    #         "严格禁止：空格、标点、连字符（-）、代码块（```）、逐字分行或任何解释性文字。"
    #     ),
    #     "meta": {"down": down, "blur_low": blur_low, "hp_radius": hp_radius, "hp_amount": hp_amount}
    # }

    # return {
    #     "outs": [os.path.abspath(p_low), os.path.abspath(p_hi)],
    #     "prompts": [
    #         # 低频视图（缩小+模糊）：读大尺度字形，忽略细节纹理
    #         (
    #                     "Identify the content in the picture.\n"
    #                     "Strict output rules:\n"
    #                     "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
    #                     "- No explanations, no punctuation, no quotes, no tags.\n"
    #                     "- If any letter is uncertain, replace it with '?'.\n"
    #                     "Now output only the word."
    #         ),
    #         # 高频视图（USM 锐化）：读细节边缘，补全笔画与断裂
    #         (
    #                     "Identify the content in the picture.\n"
    #                     "Strict output rules:\n"
    #                     "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
    #                     "- No explanations, no punctuation, no quotes, no tags.\n"
    #                     "- If any letter is uncertain, replace it with '?'.\n"
    #                     "Now output only the word."
    #         )
    #     ],
    #     # 兜底：当客户端未消费 per-image prompts 时使用
    #     "vlm_prompt": (
    #         "Read the visible English word from the image (lowercase a–z only, no punctuation). "
    #         "If unsure about any character, use '?'. Output a single word only."
    #     ),
    #     "meta": {"down": down, "blur_low": blur_low, "hp_radius": hp_radius, "hp_amount": hp_amount}
    # }

    return {
        "outs": [os.path.abspath(p_low), os.path.abspath(p_hi)],
        "prompts": [
            # 低频视图（缩小+模糊）：读大尺度字形，忽略细节纹理
            (
                "You are viewing a LOW-FREQUENCY rendering of the image (downscaled + blurred) "
                "to emphasize large, coarse letter shapes while suppressing fine textures.\n\n"
                "Step 1 – Observe: Focus on broad, smooth letter silhouettes; ignore thin lines and noise.\n"
                "Step 2 – Infer: Integrate these coarse shapes into a plausible English word a human would read.\n"
                "Step 3 – Verify: Check that spacing and global structure look natural for a word.\n"
                "Step 4 – Output: Write only ONE lowercase English word (a–z). "
                "No spaces, punctuation, or explanations; use '?' for uncertain characters; 2–16 chars.\n\n"
                "Respond only with the word."
            ),
            # 高频视图（USM 锐化）：读细节边缘，补全笔画与断裂
            (
                "You are viewing a HIGH-FREQUENCY enhanced version of the image (unsharp mask + autocontrast) "
                "to highlight edges and fine strokes.\n\n"
                "Step 1 – Observe: Look for sharpened edges and thin strokes that define letters.\n"
                "Step 2 – Infer: Combine visible edges into a coherent English word.\n"
                "Step 3 – Verify: Ensure letter forms are consistent and the word is linguistically plausible.\n"
                "Step 4 – Output: Write only ONE lowercase English word (a–z). "
                "No spaces, punctuation, or explanations; use '?' for uncertain characters; 2–16 chars.\n\n"
                "Respond only with the word."
            )
        ],
        # 兜底：当客户端未消费 per-image prompts 时使用
        "vlm_prompt": (
            "Read the visible English word from the image (lowercase a–z only, no punctuation). "
            "If unsure about any character, use '?'. Output a single word only."
        ),
        "meta": {"down": down, "blur_low": blur_low, "hp_radius": hp_radius, "hp_amount": hp_amount}
    }



# =========================================================
# ---------- 3. DoubleLayer：幻影坦克分离两层（带 per-image CoT prompt） ----------
# =========================================================

@mcp.tool()
def tp_doublelayer_phantom(src: str, outdir: str = "", variants: str = "rgb") -> dict:
    """
    幻影坦克分层：
      - 若图像有 alpha：输出“白底显影、黑底显影”两张 RGB 图；
      - 若无 alpha：回退到 GMM + 引导滤波分层，同样只返回两张层图。
    返回：
      - outs: [路径1, 路径2]
      - prompts: [对应路径1的 CoT 提示, 对应路径2的 CoT 提示]
      - vlm_prompt: 兜底统一提示（客户端未消费 per-image prompts 时使用）
      - meta: 处理信息
    """
    import os, numpy as np
    from PIL import Image

    _ensure_dir(outdir or os.path.dirname(src))
    base = os.path.join(outdir or os.path.dirname(src),
                        os.path.splitext(os.path.basename(src))[0])

    def _to_rgba(path):
        im = Image.open(path).convert("RGBA")
        return (np.array(im).astype(np.float32) / 255.0)  # H W 4

    def _composite_over_bg(rgba, bg_val):
        rgb = rgba[..., :3]; a = rgba[..., 3:4]
        return np.clip(a * rgb + (1.0 - a) * bg_val, 0., 1.)

    def _save_rgb01(p, rgb01):
        Image.fromarray((np.clip(rgb01,0,1)*255 + 0.5).astype(np.uint8)).save(p)

    # ---------- RGBA 优先（开/关灯） ----------
    has_alpha = True
    try:
        rgba = _to_rgba(src)
    except Exception:
        has_alpha = False

    outs, meta = [], {}

    if has_alpha:
        # 开/关灯合成
        rgb_white = _composite_over_bg(rgba, 1.0)  # 白底显影
        rgb_black = _composite_over_bg(rgba, 0.0)  # 黑底显影

        p_w = f"{base}_layer_white.png"
        p_b = f"{base}_layer_black.png"
        _save_rgb01(p_w, rgb_white)
        _save_rgb01(p_b, rgb_black)

        outs = [os.path.abspath(p_w), os.path.abspath(p_b)]
        meta.update({"method":"rgba_toggle_background","variants":variants})

        prompts = [
            # layer 1: white background
            (
                    "Task：Identify the content in the picture.\n"
                    "You are viewing a phantom (double-layer) image composited on a WHITE background to reveal hidden text."
                        "Strict output rules:\n"
                        "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
                        "- No explanations, no punctuation, no quotes, no tags.\n"
                        "- If any letter is uncertain, replace it with '?'.\n"
                        "Now output only the word."
            ),
            # layer 2: black background
            (
                    "Task：Identify the content in the picture.\n"
                    "You are viewing a phantom (double-layer) image composited on a WHITE background to reveal hidden text."
                        "Strict output rules:\n"
                        "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
                        "- No explanations, no punctuation, no quotes, no tags.\n"
                        "- If any letter is uncertain, replace it with '?'.\n"
                        "Now output only the word."
            )
        ]

        # prompts = [
        #     # layer 1: white background
        #     (
        #         "You are viewing a phantom (double-layer) image composited on a WHITE background to reveal hidden text.\n"
        #         "Step 1 – Observe: Ignore background textures and focus on broad letter-like shapes.\n"
        #         "Step 2 – Infer: Integrate these shapes into a plausible English word a human would read under this view.\n"
        #         "Step 3 – Verify: Check spacing/structure for a natural word.\n"
        #         "Step 4 – Output: Write only ONE lowercase English word (a–z). "
        #         "No spaces/punctuation/explanations; use '?' for unsure; 2–16 chars.\n\n"
        #         "Respond only with the word."
        #     ),
        #     # layer 2: black background
        #     (
        #         "You are viewing a phantom (double-layer) image composited on a BLACK background to reveal hidden text.\n"
        #         "Step 1 – Observe: Ignore background textures and focus on broad letter-like shapes.\n"
        #         "Step 2 – Infer: Integrate these shapes into a plausible English word a human would read under this view.\n"
        #         "Step 3 – Verify: Check spacing/structure for a natural word.\n"
        #         "Step 4 – Output: Write only ONE lowercase English word (a–z). "
        #         "No spaces/punctuation/explanations; use '?' for unsure; 2–16 chars.\n\n"
        #         "Respond only with the word."
        #     )
        # ]

        return {
            "outs": outs,
            "prompts": prompts,  # 每张图一个 CoT 提示
            "vlm_prompt": "Read the visible English word (lowercase, a–z, no punctuation).",
            "meta": meta
        }

    # ---------- 无 alpha：回退（只返回两张层图） ----------
    # 复用现有的 GMM + guided filter，最终只输出 m1/m2 两张：
    g = _to_gray_arr(src).astype(np.uint8)
    H, W = g.shape
    if HAS_CV:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g_eq = clahe.apply(g)
        g_smooth = cv2.bilateralFilter(g_eq, d=7, sigmaColor=20, sigmaSpace=7)
    else:
        g_smooth = g

    x = g_smooth.reshape(-1,1).astype(np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _, labels, centers = cv2.kmeans(x, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        means = centers.flatten()
        vars_ = np.zeros(2, np.float32) + 1e-6
        for k in (0,1):
            xs = x[labels.flatten()==k]
            if xs.size>0: vars_[k] = float(np.var(xs)) + 1e-6
        weights = np.array([np.mean(labels==k) for k in (0,1)], np.float32) + 1e-6
    except Exception:
        thr, bw = cv2.threshold(g_smooth, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) if HAS_CV \
                  else (np.percentile(g_smooth,50), (g_smooth>np.percentile(g_smooth,50)).astype(np.uint8)*255)
        xs0 = g_smooth[bw==0].astype(np.float32); xs1 = g_smooth[bw>0].astype(np.float32)
        if xs0.size==0 or xs1.size==0:
            xs0 = g_smooth[g_smooth<=np.median(g_smooth)]
            xs1 = g_smooth[g_smooth> np.median(g_smooth)]
        means = np.array([xs0.mean(), xs1.mean()], np.float32)
        vars_  = np.array([xs0.var()+1e-6, xs1.var()+1e-6], np.float32)
        weights= np.array([xs0.size/(H*W)+1e-6, xs1.size/(H*W)+1e-6], np.float32)

    idx_bright = int(np.argmax(means))
    idx_dark   = 1 - idx_bright
    mu_b, mu_d = float(means[idx_bright]), float(means[idx_dark])
    var_b, var_d = float(vars_[idx_bright]), float(vars_[idx_dark])
    wb, wd = float(weights[idx_bright]), float(weights[idx_dark])

    x1 = x.flatten()
    log_pb = -0.5*np.log(2*np.pi*var_b) - ((x1-mu_b)**2)/(2*var_b) + np.log(wb)
    log_pd = -0.5*np.log(2*np.pi*var_d) - ((x1-mu_d)**2)/(2*var_d) + np.log(wd)
    mmax = np.maximum(log_pb, log_pd)
    pb = np.exp(log_pb - mmax); pd = np.exp(log_pd - mmax)
    post_b = (pb/(pb+pd+1e-8)).reshape(H,W).astype(np.float32)
    post_d = 1.0 - post_b

    def _box(img, r): return cv2.blur(img, (2*r+1,2*r+1))
    def guided_filter(I, p, r=4, eps=1e-3):
        mean_I = _box(I,r); mean_p = _box(p,r)
        mean_Ip = _box(I*p,r); cov_Ip = mean_Ip - mean_I*mean_p
        mean_II = _box(I*I,r); var_I = mean_II - mean_I*mean_I
        a = cov_Ip/(var_I+eps); b = mean_p - a*mean_I
        mean_a = _box(a,r); mean_b = _box(b,r)
        return np.clip(mean_a*I + mean_b, 0, 1)

    I_guide = (g_smooth/255.0).astype(np.float32)
    post_b_ref = guided_filter(I_guide, post_b, r=4, eps=1e-4)
    post_d_ref = 1.0 - post_b_ref

    if HAS_CV:
        def clean(m):
            m = (m*255).astype(np.uint8)
            m = cv2.medianBlur(m,3)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,k, iterations=1)
            return (m>127).astype(np.uint8)*255
        m1 = clean(post_b_ref); m2 = clean(post_d_ref)
    else:
        m1 = (post_b_ref>0.5).astype(np.uint8)*255
        m2 = (post_d_ref>0.5).astype(np.uint8)*255

    p1 = f"{base}_phantom_layer1.png"
    p2 = f"{base}_phantom_layer2.png"
    _save(p1, _enhance_contrast_sharp(m1))
    _save(p2, _enhance_contrast_sharp(m2))
    outs = [os.path.abspath(p1), os.path.abspath(p2)]

    # prompts = [
    #     (
    #                 "Identify the content in the picture.\n"
    #                     "Strict output rules:\n"
    #                     "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
    #                     "- No explanations, no punctuation, no quotes, no tags.\n"
    #                     "- If any letter is uncertain, replace it with '?'.\n"
    #                     "Now output only the word."
    #     ),
    #     (
    #                 "Identify the content in the picture.\n"
    #                     "Strict output rules:\n"
    #                     "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
    #                     "- No explanations, no punctuation, no quotes, no tags.\n"
    #                     "- If any letter is uncertain, replace it with '?'.\n"
    #                     "Now output only the word."
    #     )
    # ]

    prompts = [
        (
            "You are viewing LAYER 1 extracted from a phantom (double-layer) image via probabilistic decomposition.\n"
            "Step 1 – Observe: Focus on coherent letter structures; ignore residual textures.\n"
            "Step 2 – Infer: Combine visible shapes into a plausible English word.\n"
            "Step 3 – Verify: Ensure spacing/structure looks like a real word.\n"
            "Step 4 – Output: Write only ONE lowercase English word (a–z). "
            "No spaces/punctuation/explanations; use '?' for unsure; 2–16 chars.\n\n"
            "Respond only with the word."
        ),
        (
            "You are viewing LAYER 2 extracted from a phantom (double-layer) image via probabilistic decomposition.\n"
            "Step 1 – Observe: Focus on coherent letter structures; ignore residual textures.\n"
            "Step 2 – Infer: Combine visible shapes into a plausible English word.\n"
            "Step 3 – Verify: Ensure spacing/structure looks like a real word.\n"
            "Step 4 – Output: Write only ONE lowercase English word (a–z). "
            "No spaces/punctuation/explanations; use '?' for unsure; 2–16 chars.\n\n"
            "Respond only with the word."
        )
    ]

    return {
        "outs": outs,
        "prompts": prompts,  # 每张图一个 CoT 提示
        "vlm_prompt": "Read the visible English word (lowercase, a–z, no punctuation).",
        "meta": {
            "method":"no_alpha_fallback_gmm",
            "mu_bright": mu_b, "mu_dark": mu_d,
            "var_bright": var_b, "var_dark": var_d,
            "w_bright": wb, "w_dark": wd
        }
    }


def _build_wedge_mask(shape, peaks, bw_deg=10, rmin=6, rmax_ratio=0.9):
    H, W = shape
    cy, cx = H//2, W//2
    yy, xx = np.ogrid[:H, :W]
    ry, rx = yy - cy, xx - cx
    rad = np.sqrt(rx*rx + ry*ry)
    ang = np.rad2deg(np.arctan2(ry, rx))
    rmax = rmax_ratio * min(H, W) * 0.5
    mask = np.ones((H, W), np.float32)

    def add_wedge(theta):
        # 对称三个方向：θ, θ±180°
        for th in (theta, theta+180, theta-180):
            da = np.abs((ang - th + 180) % 360 - 180)
            cond = (da <= bw_deg) & (rad >= rmin) & (rad <= rmax)
            mask[cond] = 0.0

    # 从峰估角度；多峰时逐个扇区挖掉
    for (r, c) in peaks:
        theta = np.rad2deg(math.atan2(r - cy, c - cx))
        add_wedge(theta)
    return mask

def _estimate_fft_peaks(mag, ring, topk=8, nms=11):
    H, W = mag.shape
    cy, cx = H//2, W//2
    yy, xx = np.ogrid[:H, :W]
    work = mag.copy()
    work[(yy - cy) ** 2 + (xx - cx) ** 2 <= ring * ring] = 0.0
    maxf = cv2.dilate(work.astype(np.float32), np.ones((nms, nms), np.uint8))
    pts = np.argwhere((work == maxf) & (work > 0))
    if pts.size == 0:
        return []
    vals = work[pts[:, 0], pts[:, 1]]
    idx = np.argsort(-vals)[:topk]
    sel = [(int(pts[i, 0]), int(pts[i, 1])) for i in idx]
    return sel

def _postprocess_for_ocr(gray_u8):
    # 先做对比度 + 轻锐化
    im = Image.fromarray(gray_u8)
    im = ImageOps.autocontrast(im, cutoff=1)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2))
    g = np.array(im)

    # CLAHE（OpenCV）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)

    # 自适应二值 + 轻微开运算去噪
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return g, bw


def _striped_vlm_prompt() -> str:
    # 条纹错觉专用、强约束的 VLM 提示词
    return (
                        "Identify the content in the picture.\n"
                        "Strict output rules:\n"
                        "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
                        "- No explanations, no punctuation, no quotes, no tags.\n"
                        "- If any letter is uncertain, replace it with '?'.\n"
                        "Now output only the word."
    )

    # return (
    #     "You are an OCR engine for striped-illusion text that has been pre-denoised.\n"
    #     "Output ONLY ONE answer.\n"
    #     "Rules: lowercase letters a-z only; use '?' for unsure characters; single token 2-16 chars; "
    #     "no spaces/punctuation/quotes; do not add any other text.\n"
    #     "now output:"
    # )

@mcp.tool()
def tp_striped_notch(
    src: str,
    outdir: str = "",
    # 你确认过的默认值
    ring: int = 60,
    peaks: int = 10,
    bw: int = 26,
    harm: int = 2,
    wedge_bw_deg: float = 14.0,
    debug: bool = False,
    return_bin: bool = False,     # ★ 默认不返回二值图
) -> dict:
    """
    条纹错觉：频域自适应陷波 + 方向楔形补刀 + 轻增强（默认仅返回灰度去纹图）
    """
    base_dir = outdir or os.path.dirname(src)
    os.makedirs(base_dir, exist_ok=True)
    base = os.path.join(base_dir, os.path.splitext(os.path.basename(src))[0])

    # 读灰度
    g = _to_gray_arr(src)

    # 第 1 段：自适应陷波（含谐波）
    viz = f"{base}_notch_spectrum.png" if debug else None
    den1 = _fft_notch(g, ring=ring, peaks=peaks, bw=bw, harm=harm, viz_path=viz)

    # 第 2 段：楔形带阻（用 den1 的频谱估角度，整扇区抑制残留条纹）
    F = np.fft.fftshift(np.fft.fft2(den1.astype(np.float32)))
    mag = np.abs(F)
    short = min(den1.shape[:2])
    ring_adapt = ring if ring is not None else max(10, int(0.04 * short))
    peak_pts = _estimate_fft_peaks(mag, ring=ring_adapt, topk=peaks, nms=11)
    wedge = _build_wedge_mask(mag.shape, peak_pts, bw_deg=wedge_bw_deg, rmin=6, rmax_ratio=0.92)
    F2 = F * wedge
    den2 = np.real(np.fft.ifft2(np.fft.ifftshift(F2)))
    den2 -= den2.min()
    if den2.max() > 0:
        den2 = den2 / den2.max() * 255.0
    den2 = den2.astype(np.uint8)

    # 轻锐化的灰度去纹版本（保持字形，不猛锐化）
    out_gray = _enhance_contrast_sharp(den2, sharp=1.3)
    p_gray = f"{base}_notch.png"
    _save(p_gray, out_gray)

    outs = [os.path.abspath(p_gray)]

    # 仅在需要时才算/返回二值图，避免客户端收到两份答案
    if return_bin or debug:
        _, binimg = _postprocess_for_ocr(den2)
        p_bin = f"{base}_notch_bin.png"
        _save(p_bin, binimg)
        if return_bin:
            outs.append(os.path.abspath(p_bin))

    if debug and viz:
        outs.append(os.path.abspath(viz))

    return {
        "outs": outs,                       # ★ 现在默认只有一张
        "vlm_prompt": _striped_vlm_prompt(),
        "meta": {
            "ring": ring_adapt, "peaks": peaks, "bw": bw,
            "harm": harm, "wedge_bw_deg": wedge_bw_deg,
            "return_bin": return_bin
        }
    }


@mcp.tool()
def tp_direction_autocompress(src: str, outdir: str = "", ratios: str = "1.5,2,2.5,3,4,5,6,7,8,9,10,12,14,16") -> dict:
    """方向错觉：自动纵横压缩"""
    _ensure_dir(outdir or os.path.dirname(src))
    base = os.path.join(outdir or os.path.dirname(src),
                        os.path.splitext(os.path.basename(src))[0])
    g = _to_gray_arr(src)
    H, W = g.shape
    if H / W >= 1.8:
        hint = "v"
    elif W / H >= 1.8:
        hint = "h"
    else:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        hint = "v" if (gy * gy).mean() < (gx * gx).mean() else "h"
    axes = [hint] + (["h", "v"] if hint == "v" else ["v", "h"])
    cand = [float(x) for x in ratios.split(",") if x.strip()]

    def comp(gray, axis, r):
        h, w = gray.shape
        if axis == "v":
            nh = max(1, int(round(h / r)))
            return cv2.resize(gray, (w, nh), interpolation=cv2.INTER_CUBIC)
        else:
            nw = max(1, int(round(w / r)))
            return cv2.resize(gray, (nw, h), interpolation=cv2.INTER_CUBIC)

    def score(gray):
        inv = 255 - gray
        _, bw = cv2.threshold(inv, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, _, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
        area = gray.size
        s = 0.0
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < 0.0005 * area:
                continue
            ar = w / max(1, h)
            if ar < 0.2 or ar > 3.5:
                continue
            s += 1.0
        return s

    best = None
    for ax in axes:
        for r in cand:
            im = comp(g, ax, r)
            s = score(im)
            if (best is None) or (s > best["score"]):
                best = {"axis": ax, "ratio": r, "img": im, "score": s}

    out = _enhance_contrast_sharp(best["img"])
    p = f"{base}_compress.png"
    _save(p, out)
    return {
        "outs": [os.path.abspath(p)],
        "vlm_prompt":   
                    "Task:Identify the content in the picture."
                    "Strict output rules:\n"
                    "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
                    "- No explanations, no punctuation, no quotes, no tags.\n"
                    "- If any letter is uncertain, replace it with '?'.\n"
                    "Now output only the word.",
        "meta": best
    }
    # return {
    #     "outs": [os.path.abspath(p)],
    #     "vlm_prompt": (
    #         "You are viewing an illusion image that originally appeared stretched or compressed "
    #         "in one direction. It has been automatically rescaled to restore its natural reading ratio.\n\n"
    #         "Step 1 – Observe: Examine the corrected image and identify regions where letters now appear "
    #         "proportionate and horizontally aligned.\n"
    #         "Step 2 – Infer: Integrate these shapes into a coherent English word that a human would read "
    #         "after the distortion has been corrected.\n"
    #         "Step 3 – Verify: Check that the letters are evenly spaced and the word looks linguistically natural.\n"
    #         "Step 4 – Output: Write only that word in lowercase English letters (a–z). "
    #         "Do not include spaces, punctuation, or explanations. "
    #         "If any character is uncertain, replace it with '?'.\n\n"
    #         "Respond only with the final word."
    #     ),
    #     "meta": best
    # }

@mcp.tool()
def tp_illusory_outline(src: str, outdir: str = "") -> dict:
    print("tp_illusory_outline")
    """虚幻轮廓：去噪+锐化"""
    _ensure_dir(outdir or os.path.dirname(src))
    base = os.path.join(outdir or os.path.dirname(src),
                        os.path.splitext(os.path.basename(src))[0])
    # den = src
    g = _to_gray_arr(src)
    den = cv2.medianBlur(g, 3)
    den = _enhance_contrast_sharp(den, sharp=1.5)
    p = f"{base}_illusory.png"
    _save(p, den)
    return {
        "outs": [os.path.abspath(p)],
        "vlm_prompt": (
            "You are analyzing an *illusory-outline* English word image. "
            "The image may have fragmented or occluded strokes.\n\n"
            "Use human-like perceptual reasoning and Gestalt closure to reconnect broken strokes, "
            "ignore background textures, and reconstruct the hidden word.\n\n"
            # "Steps:\n"
            # "1) Observe globally; ignore non-letter textures.\n"
            # "2) Complete shapes (close loops, extend stems/crossbars) while preserving symmetry/curvature.\n"
            # "3) Identify letter features; integrate into one plausible English word.\n"
            # "4) Output exactly one lowercase English word (a–z), 2–16 letters; "
            # "no punctuation/explanations; use '?' when unsure.\n\n"
            "Respond only with the final word."
            "Strict output rules:\n"
            "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
            "- No explanations, no punctuation, no quotes, no tags.\n"
            "Now output only the word."
        ),
        "meta": {}
    }

    # return {
    #     "outs": [os.path.abspath(p)],
    #     "vlm_prompt": (
    #         "You are analyzing an *illusory-outline* English word image. "
    #         "The image has been denoised (median blur) and lightly sharpened with autocontrast, "
    #         "so some strokes may appear fragmented, faded, or occluded.\n\n"

    #         "Use human-like perceptual reasoning to reconstruct the hidden English word.\n\n"

    #         "Follow these reasoning steps carefully in your mind:\n"
    #         "Step 1 – Observe Globally:\n"
    #         "• Look at the entire image holistically; identify the rhythm and alignment of shapes.\n"
    #         "• Ignore textures, patterns, and background noise—they are not letters.\n\n"

    #         "Step 2 – Complete the Shape (Gestalt Closure):\n"
    #         "• Mentally connect broken or missing strokes.\n"
    #         "• Close partial loops (a, b, d, o, p, q) and extend incomplete stems or crossbars.\n"
    #         "• Preserve the natural symmetry and curvature of letters.\n\n"

    #         "Step 3 – Identify Letter Features:\n"
    #         "• Compare each visible fragment to typical lowercase English letters:\n"
    #         "  a: right stem + small loop\n"
    #         "  b: tall stem + round belly on right\n"
    #         "  c: semicircle open to the right\n"
    #         "  d: tall stem + round belly on left\n"
    #         "  e: small loop with a middle bar\n"
    #         "  f: tall stem + small hook, mid crossbar\n"
    #         "  g: closed or open lower loop with tail\n"
    #         "  h: tall stem + arch; right curve may fade\n"
    #         "  i: short vertical stroke with dot (may be missing)\n"
    #         "  j: like 'i' but descends below baseline\n"
    #         "  k: stem + angled arm meeting at mid height\n"
    #         "  l: simple vertical stroke\n"
    #         "  m: three arches connected by stems\n"
    #         "  n: two arches connected by stems\n"
    #         "  o: round closed loop\n"
    #         "  p: stem + small loop below baseline\n"
    #         "  q: loop + small tail to bottom right\n"
    #         "  r: short stem + small curved arm\n"
    #         "  s: smooth double curve\n"
    #         "  t: short stem + top crossbar\n"
    #         "  u: cup-shaped, open upward\n"
    #         "  v: two diagonal lines meeting at bottom\n"
    #         "  w: double 'v' shape\n"
    #         "  x: two diagonals crossing\n"
    #         "  y: 'v' top + descending tail\n"
    #         "  z: two horizontals connected by diagonal\n\n"

    #         "Step 4 – Integrate:\n"
    #         "• Combine recognized letters into a single, plausible English word.\n"
    #         "• If uncertain about any letter, replace it with '?'.\n"
    #         "• Ensure spacing and shape rhythm look natural.\n\n"

    #         "Step 5 – Final Decision:\n"
    #         "• You must output exactly **one lowercase English word** (2–16 letters).\n"
    #         "• No reasoning, no explanation, no quotes, no punctuation, no commentary.\n"
    #         "• Output only the word itself. Nothing else.\n\n"

    #         "Now, think through all steps in your mind, and respond with only the final word — "
    #         "a single lowercase English word with no additional text."
    #     ),
    #     "meta": {}
    # }



@mcp.tool()
def tp_normal_unsafe(src: str, outdir: str = "") -> dict:
    """普通不安全词：轻增强"""
    _ensure_dir(outdir or os.path.dirname(src))
    base = os.path.join(outdir or os.path.dirname(src),
                        os.path.splitext(os.path.basename(src))[0])
    g = _to_gray_arr(src)
    out = _enhance_contrast_sharp(g)
    p = f"{base}_unsafe.png"
    _save(p, out)
    # return {
    #     "outs": [os.path.abspath(p)],
    #     "vlm_prompt":    "Identify the content in the picture.\n"
    #                     "Strict output rules:\n"
    #                     "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
    #                     "- No explanations, no punctuation, no quotes, no tags.\n"
    #                     "- If any letter is uncertain, replace it with '?'.\n"
    #                     "Now output only the word.",
    #     "meta": {}
    # }
    return {
        "outs": [os.path.abspath(p)],
        "vlm_prompt":     "You will read a image that exposes a hidden English word.\n"
                        "Think through the reading in your mind, but output only the final word.\n\n"
                        "Strict output rules:\n"
                        "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
                        "- No explanations, no punctuation, no quotes, no tags.\n"
                        "- If any letter is uncertain, replace it with '?'.\n"
                        "Now output only the word.",
        "meta": {}
    }

# =========================================================
# ---------- 启动 ----------
# =========================================================
if __name__ == "__main__":
    mcp.run(transport="stdio")
