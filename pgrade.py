#!/usr/bin/env python3
"""
pgrade - one-step photo colour grading with AI masks.

  pgrade.py analyze IMG...                     scene report + suggested looks (JSON)
  pgrade.py grade   IMG... [--look auto]       grade; "auto" picks a look per photo
  pgrade.py sheet   IMG [--looks a,b,c|all]    one photo rendered in many looks
  pgrade.py masks   IMG                        preview the AI masks
  pgrade.py looks                              list the looks
  pgrade.py lut     --look NAME OUT.cube       export a look's global colour as a 3D LUT

Grading runs on full-resolution float32 data. Masks (sky, person, face, skin, subject)
come from small ONNX/ncnn models at proxy resolution and are refined to full
resolution with an edge-aware colour guided filter, so edges follow the photo.
"""
import argparse, colorsys, copy, glob, json, math, os, sys, time, urllib.request
import numpy as np
import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass
from PIL import Image, ImageOps, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("PGRADE_CACHE", os.path.expanduser("~/.cache/photo-grade"))
LOOK_DIRS = [os.path.join(HERE, "looks"), os.path.expanduser("~/.config/photo-grade/looks")]
PROXY = 1024  # long side used for masks and analysis

_SKY = "https://raw.githubusercontent.com/xiongzhu666/Sky-Segmentation-and-Post-processing/main/"
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {  # all small except subject (~170 MB, only fetched when a look needs it)
    "sky_param": _SKY + "skysegsmall_sim-opt-fp16.param",
    "sky_bin": _SKY + "skysegsmall_sim-opt-fp16.bin",
    "person": _ZOO + "human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx",
    "face": _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "subject": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def model_path(key):
    url = MODELS[key]
    fn = os.path.join(CACHE, os.path.basename(url))
    if not os.path.exists(fn):
        os.makedirs(CACHE, exist_ok=True)
        log(f"[pgrade] downloading {key} model ...")
        tmp = fn + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception:
            if os.system(f'curl -sSL -o "{tmp}" "{url}"') != 0:
                raise
        os.replace(tmp, fn)
    return fn


# =============================================================== colour maths
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M1i = np.linalg.inv(_M1).astype(np.float32)
_M2i = np.linalg.inv(_M2).astype(np.float32)
_S2L = (lambda v: np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4))(
    np.linspace(0, 1, 8192, dtype=np.float64)).astype(np.float32)


def luma(x):
    return x @ LUMA


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def lut1d(x, table):
    n = len(table)
    return table[np.clip(x * (n - 1) + 0.5, 0, n - 1).astype(np.int32)]


def srgb_to_lin(x):
    return lut1d(x, _S2L)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def lin_to_oklab(lin):
    return np.cbrt(np.maximum(lin @ _M1.T, 0)) @ _M2.T


def oklab_to_lin(lab):
    return ((lab @ _M2i.T) ** 3) @ _M1i.T


def to_oklab(x):
    return lin_to_oklab(srgb_to_lin(x))


def angdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def oklab_to_srgb(lab):
    """Oklab -> sRGB. Out-of-gamut pixels get the largest chroma that fits (found per pixel by
    bisection), keeping hue and lightness. Continuous, so no contour bands in skies."""
    lin = oklab_to_lin(lab)
    bad = (lin.min(-1) < -1e-3) | (lin.max(-1) > 1.001)
    if bad.any():
        sub = lab[bad]
        lo, hi = np.zeros(len(sub), np.float32), np.ones(len(sub), np.float32)
        for _ in range(9):
            mid = (lo + hi) / 2
            t = sub.copy()
            t[:, 1:] *= mid[:, None]
            l = oklab_to_lin(t)
            ok = (l.min(-1) >= -1e-3) & (l.max(-1) <= 1.001)
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        sub[:, 1:] *= lo[:, None]
        lin[bad] = oklab_to_lin(sub)
    return lin_to_srgb(lin)


def wheel_dir(hue):
    """Lightroom colour-wheel hue (HSV degrees) -> unit direction in Oklab a,b."""
    rgb = np.array(colorsys.hsv_to_rgb((hue % 360) / 360.0, 1, 1), np.float32)
    v = rgb - rgb @ LUMA  # luminance-free push toward the hue, measured at mid grey (Lightroom-like)
    ab = (to_oklab(np.clip(0.5 + 0.1 * v, 0, 1)[None]) - to_oklab(np.full((1, 3), 0.5, np.float32)))[0, 1:]
    return ab / (np.linalg.norm(ab) + 1e-9)


# ============================================================ guided filters
def box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_gray(I, p, r, eps):
    mI, mp = box(I, r), box(p, r)
    a = (box(I * p, r) - mI * mp) / (box(I * I, r) - mI * mI + eps)
    b = mp - a * mI
    return box(a, r) * I + box(b, r)


def guided_coeffs(I, p, r, eps):
    """Colour-guide guided filter (He et al.) -> averaged coefficients (A, b)."""
    mI, mp = box(I, r), box(p, r)
    cov = box(I * p[..., None], r) - mI * mp[..., None]
    c = lambda i, j: box(I[..., i] * I[..., j], r) - mI[..., i] * mI[..., j]
    rr, rg, rb, gg, gb, bb = c(0, 0) + eps, c(0, 1), c(0, 2), c(1, 1) + eps, c(1, 2), c(2, 2) + eps
    i_rr, i_rg, i_rb = gg * bb - gb * gb, gb * rb - rg * bb, rg * gb - gg * rb
    i_gg, i_gb, i_bb = rr * bb - rb * rb, rb * rg - rr * gb, rr * gg - rg * rg
    det = rr * i_rr + rg * i_rg + rb * i_rb
    A = np.stack([(i_rr * cov[..., 0] + i_rg * cov[..., 1] + i_rb * cov[..., 2]) / det,
                  (i_rg * cov[..., 0] + i_gg * cov[..., 1] + i_gb * cov[..., 2]) / det,
                  (i_rb * cov[..., 0] + i_gb * cov[..., 1] + i_bb * cov[..., 2]) / det], -1)
    b = mp - (A * mI).sum(-1)
    return box(A.astype(np.float32), r), box(b.astype(np.float32), r)


def guided_color(I, p, r, eps):
    A, b = guided_coeffs(I, p, r, eps)
    return (A * I).sum(-1) + b


# ================================================================ tone stages
TONE_KEYS = ("contrast", "highlights", "shadows", "whites", "blacks")
K_TONE, K_HIGHLIGHTS, K_WHITES, K_CONTRAST = 0.30, 0.22, 0.60, 0.80


def tone_table(P, n=4096):
    c, h, s, w, b = (P.get(k, 0) / 100 for k in TONE_KEYS)
    y = np.linspace(0, 1, n, dtype=np.float32)
    y = y - K_CONTRAST * c * np.sin(2 * np.pi * y) / (2 * np.pi)
    y = y + K_TONE * s * 6.75 * y * (1 - y) ** 2
    y = y + K_HIGHLIGHTS * h * 12.2 * y ** 4 * (1 - y)
    y = y + K_WHITES * w * y ** 3
    y = y + 0.08 * b * (1 - y) ** 6
    return np.maximum.accumulate(np.clip(y, 0, 1)).astype(np.float32)


def st_tone(x, P):
    if not any(P.get(k) for k in TONE_KEYS):
        return x
    t = tone_table(P)
    L = luma(x)
    ratio = (lut1d(L, t) + 1e-4) / (L + 1e-4)
    return np.clip(0.55 * x * ratio[..., None] + 0.45 * lut1d(x, t), 0, 1)


def curve_table(pts, n=4096):
    """Monotone cubic (Fritsch-Carlson) through Lightroom-style 0..255 points."""
    p = np.array(sorted(pts), np.float64) / 255.0
    xs, ys = p[:, 0], p[:, 1]
    h, d = np.diff(xs), np.diff(ys) / np.diff(xs)
    m = np.empty_like(ys)
    m[0], m[-1] = d[0], d[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        m[1:-1] = np.where(d[:-1] * d[1:] > 0, 2 / (1 / d[:-1] + 1 / d[1:]), 0)
    X = np.linspace(0, 1, n)
    i = np.clip(np.searchsorted(xs, X) - 1, 0, len(xs) - 2)
    t = np.clip((X - xs[i]) / h[i], 0, 1)
    Y = ((2 * t**3 - 3 * t**2 + 1) * ys[i] + (t**3 - 2 * t**2 + t) * h[i] * m[i]
         + (-2 * t**3 + 3 * t**2) * ys[i + 1] + (t**3 - t**2) * h[i] * m[i + 1])
    Y = np.where(X < xs[0], ys[0], np.where(X > xs[-1], ys[-1], Y))
    return np.clip(Y, 0, 1).astype(np.float32)


def st_curves(x, P):
    if P.get("curve"):
        x = lut1d(x, curve_table(P["curve"]))
    for i, k in enumerate(("curve_r", "curve_g", "curve_b")):
        if P.get(k):
            x[..., i] = lut1d(x[..., i], curve_table(P[k]))
    return x


def st_linear(x, P, gains=None):
    """White balance (temperature/tint, JPEG-style +-100) and exposure in linear light."""
    t, ti, ev = P.get("temperature", 0) / 100, P.get("tint", 0) / 100, P.get("exposure", 0)
    g = np.array([1 + 0.30 * t, 1 - 0.18 * ti, 1 - 0.30 * t], np.float32)
    g /= g @ LUMA
    if gains is not None:
        g = g * gains
    g = g * np.float32(2.0 ** ev)
    if np.allclose(g, 1, atol=1e-4):
        return x
    lin = srgb_to_lin(x) * g
    if g.max() > 1:  # soft shoulder instead of hard clipping
        k = 0.8
        lin = np.where(lin > k, k + (1 - k) * np.tanh((lin - k) / (1 - k)), lin)
    return lin_to_srgb(lin)


def st_local_contrast(x, clarity, texture):
    if not clarity and not texture:
        return x
    L = luma(x)
    out = L.copy()
    s = min(x.shape[:2])
    if clarity:
        base = guided_gray(L, L, max(4, int(s * 0.025)), 0.012)
        out += (clarity / 100) * 1.1 * (L - base) * (0.25 + 3.0 * L * (1 - L))
    if texture:
        base = guided_gray(L, L, max(2, int(s * 0.004)), 0.0015)
        out += (texture / 100) * 1.0 * (L - base)
    ratio = np.clip((np.clip(out, 0, 1) + 1e-3) / (L + 1e-3), 0.5, 2.0)
    return np.clip(x * ratio[..., None], 0, 1)


def st_dehaze(x, amount):
    if not amount:
        return x
    H, W = x.shape[:2]
    sc = min(1.0, 640 / max(H, W))
    p = cv2.resize(x, (max(1, int(W * sc)), max(1, int(H * sc))), interpolation=cv2.INTER_AREA)
    ker = np.ones((15, 15), np.uint8)
    dark = cv2.erode(p.min(-1), ker)
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = np.maximum(p.reshape(-1, 3)[idx].mean(0), 0.25).astype(np.float32)
    tr = 1 - 0.95 * cv2.erode((p / A).min(-1).astype(np.float32), ker)
    tr = guided_gray(luma(p), tr.astype(np.float32), 20, 1e-3)
    tr = cv2.resize(tr, (W, H), interpolation=cv2.INTER_LINEAR)
    a = amount / 100
    if a > 0:
        t = np.clip(tr, 0.3, 1)[..., None]
        out = x + a * 0.7 * (((x - A) / t + A) - x)
    else:
        out = x + (-a) * 0.6 * (A - x) * (0.4 + 0.6 * np.clip(1 - tr, 0, 1))[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


# ======================================================= colour (Oklab) stage
HSL_BANDS = [("red", 25), ("orange", 58), ("yellow", 100), ("green", 140),
             ("aqua", 195), ("blue", 255), ("purple", 295), ("magenta", 335)]
K_HSL_HUE, K_HSL_LUM, K_BW, K_GRADE, K_GRADE_LUM = 25.0, 0.35, 0.6, 0.075, 0.15
COLOR_KEYS = ("vibrance", "saturation", "hsl", "color_grade", "bw", "bw_mix")


def hsl_weights(h):
    cs = [c for _, c in HSL_BANDS]
    W = []
    for i, c in enumerate(cs):
        prev = cs[i - 1] - (360 if i == 0 else 0)
        nxt = cs[(i + 1) % len(cs)] + (360 if i == len(cs) - 1 else 0)
        d = angdiff(h, c)
        W.append(np.clip(np.where(d < 0, 1 + d / (c - prev), 1 - d / (nxt - c)), 0, 1))
    return np.stack(W, -1).astype(np.float32)


def st_color(x, P):
    if not any(P.get(k) for k in COLOR_KEYS):
        return x
    lab = to_oklab(x)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360
    vib, sat = P.get("vibrance", 0) / 100, P.get("saturation", 0) / 100
    if vib or sat:
        skin = np.exp(-(angdiff(h, 60) / 25) ** 2)
        k = (1 + vib * (1 - np.clip(C / 0.16, 0, 1)) ** 2 * (1 - 0.6 * skin)) * (1 + sat)
        C = C * np.maximum(k, 0)
    hsl, bw = P.get("hsl"), P.get("bw")
    if hsl or bw:
        W = hsl_weights(h)
        csat = np.clip(C / 0.06, 0, 1)
        if hsl:
            tab = np.array([hsl.get(n, [0, 0, 0]) for n, _ in HSL_BANDS], np.float32) / 100
            h = h + (W @ tab[:, 0]) * K_HSL_HUE * csat
            C = C * np.clip(1 + W @ tab[:, 1], 0, None)
            L = L * (1 + K_HSL_LUM * (W @ tab[:, 2]) * csat)
        if bw:
            mix = P.get("bw_mix", {})
            vec = np.array([mix.get(n, 0) for n, _ in HSL_BANDS], np.float32) / 100
            L = L * (1 + K_BW * (W @ vec) * csat)
            C = C * 0
    hr = np.radians(h)
    a, b = C * np.cos(hr), C * np.sin(hr)
    cg = P.get("color_grade")
    if cg:
        bal = cg.get("balance", 0) / 100 if isinstance(cg.get("balance", 0), (int, float)) else 0
        Lz = np.clip(L - 0.25 * bal, 0, 1)
        zones = {"shadows": (1 - Lz) ** 2, "midtones": 2 * Lz * (1 - Lz), "highlights": Lz ** 2,
                 "global": 0.25 + 0.6 * np.clip(L, 0, 1)}
        Lout = L.copy()
        for zone, w in zones.items():
            hue, s, lum = (list(cg.get(zone, [0, 0, 0])) + [0, 0, 0])[:3]
            if s:
                d = wheel_dir(hue) * (K_GRADE * s / 100)
                a = a + w * d[0]
                b = b + w * d[1]
            if lum:
                Lout = Lout + w * K_GRADE_LUM * lum / 100
        L = Lout
    return oklab_to_srgb(np.stack([L, a, b], -1).astype(np.float32))


# ========================================================== finishing stages
def st_bloom(x, B):
    if not B or not B.get("amount"):
        return x
    H, W = x.shape[:2]
    thr = B.get("threshold", 0.7)
    lin = srgb_to_lin(x)
    src = lin * smoothstep(thr - 0.15, thr + 0.1, luma(x))[..., None]
    f = 4
    small = cv2.resize(src, (max(1, W // f), max(1, H // f)), interpolation=cv2.INTER_AREA)
    sig = max(1.0, B.get("radius", 0.03) * max(H, W) / f)
    bl = cv2.GaussianBlur(small, (0, 0), sig) + 0.6 * cv2.GaussianBlur(small, (0, 0), sig * 0.3)
    bl = cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR) * np.array(B.get("tint", [1, 1, 1]), np.float32)
    lin = 1 - (1 - lin) * (1 - np.clip(B["amount"] * bl, 0, 1))
    return lin_to_srgb(lin)


def st_vignette(x, V):
    if not V or not V.get("amount"):
        return x
    H, W = x.shape[:2]
    yy, xx = np.ogrid[:H, :W]
    d = np.sqrt((((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2).astype(np.float32)
    start = 0.2 + 0.5 * V.get("midpoint", 50) / 100
    w = smoothstep(start, start + 0.3 + 0.6 * V.get("feather", 50) / 100, d)
    amt = V["amount"] / 100
    lin = srgb_to_lin(x)
    lin = lin * (1 + 0.85 * amt * w)[..., None] if amt < 0 else lin + (1 - lin) * (0.6 * amt * w)[..., None]
    return lin_to_srgb(lin)


def st_grain(x, G, seed=7):
    if not G or not G.get("amount"):
        return x
    H, W = x.shape[:2]
    s = max(1.0, G.get("size", 1.0) * max(H, W) / 3000)
    n = np.random.default_rng(seed).standard_normal((int(H / s) + 2, int(W / s) + 2)).astype(np.float32)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    n /= n.std() + 1e-6
    L = luma(x)
    w = 0.35 + 0.65 * (1 - (2 * L - 1) ** 2)
    return np.clip(x + (G["amount"] / 100 * 0.045 * n * w)[..., None], 0, 1)


def st_sharpen(x, amount):
    if not amount or amount <= 0:
        return x
    H, W = x.shape[:2]
    L = luma(x)
    d = L - cv2.GaussianBlur(L, (0, 0), max(0.8, max(H, W) / 3000))
    m = smoothstep(0.004, 0.03, np.abs(d))
    return np.clip(x + (amount / 100 * 1.4 * d * m)[..., None], 0, 1)


def st_chroma_denoise(x, amount=1.0):
    """Edge-aware smoothing of colour (Oklab a/b) only: stops JPEG/sensor colour noise from
    turning into blotches once saturation and colour grades push it."""
    if amount <= 0:
        return x
    lab = to_oklab(x)
    L = lab[..., 0]
    r = max(2, int(min(x.shape[:2]) * 0.006))
    for c in (1, 2):
        sm = guided_gray(L, lab[..., c], r, 4e-4)
        lab[..., c] += amount * (sm - lab[..., c])
    return oklab_to_srgb(lab)


def st_protect_skin(x, ref, m, k):
    """Keep graded tone, but pull skin hue/chroma back toward natural."""
    if m is None or k <= 0 or m.max() < 0.05:
        return x
    ys, xs = np.where(m > 0.02)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lx, lr = to_oklab(x[y0:y1, x0:x1]), to_oklab(ref[y0:y1, x0:x1])
    w = (k * m[y0:y1, x0:x1])
    hx, hr = np.degrees(np.arctan2(lx[..., 2], lx[..., 1])), np.degrees(np.arctan2(lr[..., 2], lr[..., 1]))
    Cx, Cr = np.hypot(lx[..., 1], lx[..., 2]), np.hypot(lr[..., 1], lr[..., 2])
    h = np.radians(hx + w * angdiff(hr, hx))
    C = Cx + w * (np.clip(Cx, Cr * 0.85, Cr * 1.2) - Cx)
    lx[..., 1], lx[..., 2] = C * np.cos(h), C * np.sin(h)
    out = x.copy()
    out[y0:y1, x0:x1] = oklab_to_srgb(lx)
    return out


# ================================================================ the scene
class Scene:
    """Holds an image plus lazily computed analysis and AI masks."""

    def __init__(self, img, use_ml=True):
        self.img = img
        self.H, self.W = img.shape[:2]
        sc = min(1.0, PROXY / max(self.H, self.W))
        self.pw, self.ph = max(1, round(self.W * sc)), max(1, round(self.H * sc))
        self.px = cv2.resize(img, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        self.pu8 = (self.px * 255 + 0.5).astype(np.uint8)
        self.use_ml = use_ml
        self._p, self._f, self._faces, self._stats = {}, {}, None, None
        self.disabled = set()

    # ---- AI models -------------------------------------------------------
    def _ort(self, key):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        return ort.InferenceSession(model_path(key), so, providers=["CPUExecutionProvider"])

    def _refine(self, p, r_div=80, eps=1e-3, lo=0.3, hi=0.7):
        p = cv2.resize(p.astype(np.float32), (self.pw, self.ph), interpolation=cv2.INTER_CUBIC)
        p = guided_color(self.px, np.clip(p, 0, 1), max(3, min(self.pw, self.ph) // r_div), eps)
        return smoothstep(lo, hi, p).astype(np.float32)

    def _sky(self):
        try:
            import ncnn
        except ImportError:
            log("[pgrade] ncnn not installed -> no sky mask (pip install ncnn)")
            return np.zeros((self.ph, self.pw), np.float32)
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = os.cpu_count() or 2
        net.load_param(model_path("sky_param"))
        net.load_model(model_path("sky_bin"))
        small = np.ascontiguousarray(cv2.resize(self.pu8, (384, 384), interpolation=cv2.INTER_AREA))
        m = ncnn.Mat.from_pixels(small, ncnn.Mat.PixelType.PIXEL_RGB, 384, 384)
        m.substract_mean_normalize([123.675, 116.28, 103.53], [1 / 58.395, 1 / 57.12, 1 / 57.375])
        ex = net.create_extractor()
        ex.input("input.1", m)
        _, out = ex.extract("1959")
        p = np.array(out, np.float32).reshape(384, 384)
        return self._refine(p, r_div=70, eps=1e-3, lo=0.25, hi=0.75)

    def faces(self):
        if self._faces is None:
            self._faces = []
            if self.use_ml:
                det = cv2.FaceDetectorYN.create(model_path("face"), "", (self.pw, self.ph), 0.75, 0.3, 5000)
                det.setInputSize((self.pw, self.ph))
                _, f = det.detect(np.ascontiguousarray(self.pu8[..., ::-1]))
                if f is not None:
                    self._faces = [tuple(map(float, r[:4])) + (float(r[14]),) for r in f if min(r[2], r[3]) >= 10]
        return self._faces

    def _face(self):
        m = np.zeros((self.ph, self.pw), np.float32)
        for x, y, w, h, _ in self.faces():
            cv2.ellipse(m, (int(x + w / 2), int(y + h * 0.52)), (int(w * 0.48), int(h * 0.62)), 0, 0, 360, 1.0, -1)
        return self._refine(cv2.GaussianBlur(m, (0, 0), 2), r_div=150, lo=0.2, hi=0.8) if m.any() else m

    def _person(self):
        if not self.faces():  # gate on faces: the model hallucinates people in empty scenes
            return np.zeros((self.ph, self.pw), np.float32)
        x = cv2.resize(self.pu8, (192, 192), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        s = self._ort("person")
        o = s.run(None, {s.get_inputs()[0].name: x})[0][0]
        e = np.exp(o - o.max(0))
        p = e[1] / e.sum(0)
        return np.maximum(self._refine(p, r_div=60), self.proxy_mask("face"))

    def _skin(self):
        faces = self.faces()
        if not faces:
            return np.zeros((self.ph, self.pw), np.float32)
        lab = to_oklab(self.px)
        samp = np.zeros((self.ph, self.pw), np.uint8)
        for x, y, w, h, _ in faces:
            cv2.ellipse(samp, (int(x + w / 2), int(y + h * 0.58)), (max(1, int(w * 0.28)), max(1, int(h * 0.3))), 0, 0, 360, 1, -1)
        sel = (samp > 0) & (lab[..., 0] > 0.3) & (lab[..., 0] < 0.97)
        if sel.sum() < 20:
            return self.proxy_mask("face")
        ab = lab[..., 1:][sel]
        mu, cov = ab.mean(0), np.cov(ab.T) + np.eye(2) * 2e-5
        d = lab[..., 1:] - mu
        ic = np.linalg.inv(cov)
        d2 = np.einsum("...i,ij,...j->...", d, ic, d)
        p = np.clip(1 - (np.sqrt(d2) - 2.0) / 2.0, 0, 1) * smoothstep(0.15, 0.3, lab[..., 0])
        gate = cv2.dilate(self.proxy_mask("person"), np.ones((9, 9), np.uint8))
        return self._refine(p * np.maximum(gate, self.proxy_mask("face")), r_div=150, lo=0.2, hi=0.8)

    def _subject(self):
        x = cv2.resize(self.pu8, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (x / max(x.max(), 1) - 0.5).transpose(2, 0, 1)[None]
        s = self._ort("subject")
        o = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
        o = (o - o.min()) / (o.max() - o.min() + 1e-6)
        return self._refine(o, r_div=120, lo=0.35, hi=0.65)

    # ---- mask access -----------------------------------------------------
    NAMES = ("sky", "person", "face", "skin", "subject")

    def proxy_mask(self, name):
        if name not in self._p:
            if not self.use_ml or name in self.disabled:
                self._p[name] = np.zeros((self.ph, self.pw), np.float32)
            else:
                t = time.time()
                self._p[name] = getattr(self, "_" + name)()
                log(f"[pgrade] {name} mask {time.time() - t:.1f}s, coverage {self._p[name].mean():.1%}")
        return self._p[name]

    def mask(self, name):
        """Full-resolution mask, edge-refined with a fast colour guided filter."""
        if name not in self._f:
            p = self.proxy_mask(name)
            if p.max() < 0.01:
                self._f[name] = np.zeros((self.H, self.W), np.float32)
            elif (self.pw, self.ph) == (self.W, self.H):
                self._f[name] = p
            else:
                A, b = guided_coeffs(self.px, p, 2, 1e-4)
                A = cv2.resize(A, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                b = cv2.resize(b, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self._f[name] = np.clip((A * self.img).sum(-1) + b, 0, 1).astype(np.float32)
        return self._f[name]

    def expr(self, e, img=None):
        """Mask expression: tokens joined by '*', '!' inverts. Tokens: sky person face skin subject
        background all lum:LO-HI hue:LO-HI grad:top|bottom|left|right[:EXTENT] radial[:SIZE]"""
        img = self.img if img is None else img
        total = None
        for tok in e.split("*"):
            tok = tok.strip()
            inv = tok.startswith("!")
            tok = tok.lstrip("!")
            name, _, arg = tok.partition(":")
            if name in self.NAMES:
                m = self.mask(name)
            elif name == "background":
                m = 1 - self.mask("subject")
            elif name == "all":
                m = np.ones((self.H, self.W), np.float32)
            elif name == "lum":
                lo, hi = map(float, arg.split("-"))
                L = luma(img)
                m = smoothstep(lo - 0.08, lo + 0.08, L) * (1 - smoothstep(hi - 0.08, hi + 0.08, L))
            elif name == "hue":
                lo, hi = map(float, arg.split("-"))
                lab = to_oklab(img)
                hh = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
                c, wd = (lo + angdiff(hi, lo) / 2) % 360, abs(angdiff(hi, lo)) / 2
                m = (1 - smoothstep(wd, wd + 15, np.abs(angdiff(hh, c)))) * smoothstep(0.02, 0.06, np.hypot(lab[..., 1], lab[..., 2]))
            elif name == "grad":
                side, _, ext = arg.partition(":")
                ext = float(ext or 0.5)
                yy, xx = np.mgrid[:self.H, :self.W].astype(np.float32)
                t = {"top": yy / self.H, "bottom": 1 - yy / self.H, "left": xx / self.W, "right": 1 - xx / self.W}[side or "top"]
                m = 1 - smoothstep(0, ext, t)
            elif name == "radial":
                size = float(arg or 0.6)
                yy, xx = np.ogrid[:self.H, :self.W]
                d = np.sqrt(((xx - self.W / 2) / (self.W / 2)) ** 2 + ((yy - self.H / 2) / (self.H / 2)) ** 2)
                m = (1 - smoothstep(size * 0.6, size * 1.2, d)).astype(np.float32)
            else:
                raise ValueError(f"unknown mask token: {tok}")
            m = 1 - m if inv else m
            total = m if total is None else total * m
        return total.astype(np.float32)

    # ---- analysis ----------------------------------------------------------
    def stats(self):
        if self._stats:
            return self._stats
        L = luma(self.px)
        lab = to_oklab(self.px)
        C = np.hypot(lab[..., 1], lab[..., 2])
        hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
        sky = self.proxy_mask("sky")
        sf = float(sky.mean())
        s = {"size": [self.W, self.H], "median_luma": float(np.median(L)), "mean_luma": float(L.mean()),
             "p02": float(np.percentile(L, 2)), "p98": float(np.percentile(L, 98)),
             "clipped_highlights": float((L > 0.985).mean()), "crushed_shadows": float((L < 0.015).mean()),
             "mean_chroma": float(C.mean()), "sky_fraction": sf, "faces": len(self.faces()),
             "face_area": float(sum(f[2] * f[3] for f in self.faces()) / (self.pw * self.ph))}
        if sf > 0.03:
            w = sky / sky.sum()
            sa, sb = float((lab[..., 1] * w).sum()), float((lab[..., 2] * w).sum())
            s["sky"] = {"lightness": float((lab[..., 0] * w).sum()), "chroma": float(math.hypot(sa, sb)),
                        "hue": float(math.degrees(math.atan2(sb, sa)) % 360)}
        ground = sky < 0.5
        s["ground_luma"] = float(np.median(L[ground])) if ground.mean() > 0.05 else s["median_luma"]
        if sf > 0.03:
            sel = sky > 0.5
            warm = (np.abs(angdiff(hue[sel], 65)) < 50) & (C[sel] > 0.035)
            s["sky"]["warm_fraction"] = float(warm.mean()) if sel.any() else 0.0
        bright = (L > 0.6) & (sky < 0.5)
        s["warm_highlights"] = float(lab[..., 2][bright].mean()) if bright.mean() > 0.005 else 0.0
        lights = (L > 0.55) & (C > 0.09)
        s["neon_fraction"] = float((lights & ((np.abs(angdiff(hue, 335)) < 40) | (np.abs(angdiff(hue, 210)) < 30))).mean())
        s["scene"], s["suggested_looks"] = classify(s)
        self._stats = s
        return s


def classify(s):
    sky = s.get("sky")
    ground_dark = s["ground_luma"] < 0.2
    sunset = sky and sky.get("warm_fraction", 0) > 0.12
    blue_sky = sky and 190 < sky["hue"] < 310 and sky["chroma"] >= 0.025
    if s["face_area"] > 0.015:
        scene = "portrait"
    elif sunset:
        scene = "golden_hour"
    elif ground_dark and sky and 190 < sky["hue"] < 320:
        scene = "blue_hour"
    elif ground_dark:
        scene = "night_neon" if s["neon_fraction"] > 0.004 else "night"
    elif sky and sky["chroma"] < 0.035 and sky["lightness"] > 0.55 and s["sky_fraction"] > 0.08 \
            and s["warm_highlights"] < 0.035 and s["mean_chroma"] < 0.035:  # grey sky AND a grey scene
        scene = "overcast"
    elif blue_sky or (sky and s["ground_luma"] > 0.45 and s["mean_chroma"] >= 0.035):  # incl. hazy-white sky on a sunny day
        scene = "daylight"
    elif False:
        scene = "daylight"
    elif s["warm_highlights"] > 0.05:
        scene = "golden_hour"
    else:
        scene = "other"
    looks = {"portrait": ["film-warm", "natural", "classic-chrome"],
             "golden_hour": ["gta6", "golden", "teal-orange"],
             "blue_hour": ["vice-night", "neon-noir", "teal-orange"],
             "night_neon": ["neon-noir", "vice-night", "gta6"],
             "night": ["vice-night", "neon-noir", "moody"],
             "overcast": ["moody", "classic-chrome", "bw"],
             "daylight": ["teal-orange", "natural", "golden"],
             "other": ["natural", "film-warm", "classic-chrome"]}[scene]
    return scene, looks


# ================================================================ looks
def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_looks():
    raw = {}
    for d in LOOK_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(f) as fh:
                j = json.load(fh)
            raw[j.get("name", os.path.splitext(os.path.basename(f))[0])] = j
    out = {}

    def resolve(n, seen=()):
        if n in out:
            return out[n]
        j = raw[n]
        if j.get("base"):
            if j["base"] in seen:
                raise ValueError(f"look inheritance loop at {n}")
            j = deep_merge(resolve(j["base"], seen + (n,)), {k: v for k, v in j.items() if k != "base"})
        out[n] = j
        return j
    for n in raw:
        resolve(n)
    return out


def get_look(name, looks):
    if name and os.path.isfile(name):
        with open(name) as fh:
            j = json.load(fh)
        return deep_merge(looks[j["base"]], j) if j.get("base") in looks else j
    if name not in looks:
        sys.exit(f"unknown look '{name}'. Available: {', '.join(sorted(looks))}")
    return looks[name]


# ============================================================ the pipeline
def auto_params(scene, look):
    A = look.get("auto") or {}
    s = scene.stats()
    ev, gains = 0.0, None
    if A.get("exposure") and s["scene"] not in ("night", "night_neon", "blue_hour"):
        med = max(s["median_luma"], 0.02)
        ev = float(np.clip(math.log2(float(srgb_to_lin(np.float32(0.42))) / float(srgb_to_lin(np.float32(med)))) * 0.5, -0.6, 0.7))
        if ev > 0 and s["clipped_highlights"] > 0.01:
            ev *= 0.5
    if A.get("wb"):
        keep = s["scene"] in ("golden_hour", "night", "night_neon", "blue_hour")
        f = 0.15 if keep else 0.6
        lab = to_oklab(scene.px)
        L = lab[..., 0]
        g = (np.hypot(lab[..., 1], lab[..., 2]) < 0.05) & (L > 0.3) & (L < 0.92)
        g &= (scene.proxy_mask("sky") < 0.5) & (scene.proxy_mask("skin") < 0.3)
        if g.mean() > 0.005:
            m = srgb_to_lin(scene.px)[g].mean(0) + 1e-4
            gains = np.clip(((m @ LUMA) / m) ** f, 0.85, 1.2).astype(np.float32)
    return ev, gains


def develop(x, P, scene=None):
    """Global-style adjustments (also used for each local adjustment)."""
    x = st_linear(x, P, P.get("_gains"))
    x = st_dehaze(x, P.get("dehaze", 0))
    x = st_tone(x, P)
    x = st_curves(x, P)
    x = st_local_contrast(x, P.get("clarity", 0), P.get("texture", 0))
    return st_color(x, P)


def grade(img, look, scene, strength=1.0, pop=0.0, pointwise=False, seed=7):
    P = copy.deepcopy(look)
    ev, gains = (0.0, None) if pointwise else auto_params(scene, look)
    P["exposure"] = P.get("exposure", 0) + ev
    P["_gains"] = gains
    ref = st_linear(img, {"_gains": gains, "exposure": P["exposure"]}, gains) if not pointwise else img
    if pointwise:
        P["clarity"] = P["texture"] = P["dehaze"] = 0
    x = develop(st_chroma_denoise(img, P.get("color_denoise", 0.8)) if not pointwise else img, P)
    if not pointwise:
        locs = list(P.get("local", []))
        if pop and not (0.004 < scene.proxy_mask("subject").mean() < 0.75):
            log("[pgrade] no clear subject found -> --pop skipped")
            pop = 0
        if pop:
            locs += [{"mask": "subject", "adjust": {"exposure": 0.12 * pop, "clarity": 12 * pop, "vibrance": 8 * pop}},
                     {"mask": "background", "adjust": {"exposure": -0.12 * pop, "saturation": -8 * pop}}]
        for loc in locs:
            m = scene.expr(loc["mask"]) * loc.get("amount", 1.0)
            if m.max() < 0.02:
                continue
            y = develop(x, loc["adjust"])
            x = x + m[..., None] * (y - x)
        if P.get("protect_skin"):
            x = st_protect_skin(x, ref, scene.mask("skin") if scene.faces() else None, P["protect_skin"])
        x = st_bloom(x, P.get("bloom"))
        x = st_vignette(x, P.get("vignette"))
        x = st_grain(x, P.get("grain"), seed)
        x = st_sharpen(x, P.get("sharpness", 0))
    if strength != 1.0:
        x = img + (x - img) * strength
    return np.clip(x, 0, 1).astype(np.float32)


# ====================================================================== I/O
RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif"} | RAW_EXT


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    meta = {"ext": ext, "bits16": False}
    if ext in RAW_EXT:
        import rawpy
        with rawpy.imread(path) as r:
            rgb = r.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True)
        meta["bits16"] = True
        return rgb.astype(np.float32) / 65535, meta
    if ext in (".tif", ".tiff", ".png"):
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is not None and a.dtype == np.uint16 and a.ndim == 3:
            meta["bits16"] = True
            return np.ascontiguousarray(a[..., :3][..., ::-1]).astype(np.float32) / 65535, meta
    if ext in (".heic", ".heif"):
        import pillow_heif
        pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(path))
    meta["icc"] = im.info.get("icc_profile")
    try:
        ex = im.getexif()
        if 0x0112 in ex:
            ex[0x0112] = 1
        meta["exif"] = ex.tobytes()
    except Exception:
        pass
    if im.mode in ("RGBA", "LA", "PA"):
        meta["alpha"] = im.getchannel("A")
    return np.asarray(im.convert("RGB"), np.float32) / 255, meta


def save_image(path, x, meta, quality=95):
    ext = os.path.splitext(path)[1].lower()
    if meta.get("bits16") and ext in (".tif", ".tiff", ".png"):
        cv2.imwrite(path, (np.clip(x, 0, 1)[..., ::-1] * 65535 + 0.5).astype(np.uint16))
        return
    noise = np.random.default_rng(1).uniform(-0.5, 0.5, x.shape).astype(np.float32)  # dither: no sky banding
    im = Image.fromarray(np.clip(x * 255 + noise + 0.5, 0, 255).astype(np.uint8), "RGB")
    kw = {}
    if ext in (".jpg", ".jpeg", ".webp"):
        kw["quality"] = quality
        if ext != ".webp":
            kw["subsampling"] = 0 if quality >= 90 else 2
    elif meta.get("alpha") is not None:
        im.putalpha(meta["alpha"])
    if meta.get("icc"):
        kw["icc_profile"] = meta["icc"]
    if meta.get("exif") and ext in (".jpg", ".jpeg", ".webp", ".png", ".tif", ".tiff"):
        kw["exif"] = meta["exif"]
    im.save(path, **kw)


def to_pil(x):
    return Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def label(im, text, size=None):
    d = ImageDraw.Draw(im)
    size = size or max(14, im.height // 24)
    d.text((10, 8), text, fill="white", font=_font(size), stroke_width=max(2, size // 10), stroke_fill="black")
    return im


def compare_image(before, after, name):
    a, b = label(to_pil(before), "BEFORE"), label(to_pil(after), name.upper())
    w, h = a.size
    vertical = w > h * 1.15  # landscape photos stack top/bottom (reads better on phones)
    c = Image.new("RGB", (w, h * 2 + 8) if vertical else (w * 2 + 8, h), "black")
    c.paste(a, (0, 0))
    c.paste(b, (0, h + 8) if vertical else (w + 8, 0))
    return c


def collect(inputs):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p)
                            if os.path.splitext(f)[1].lower() in IMG_EXT and "_compare" not in f)
        else:
            files.append(p)
    return files


def downsize(x, long_side):
    H, W = x.shape[:2]
    s = long_side / max(H, W)
    return x if s >= 1 else cv2.resize(x, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)


# ====================================================================== CLI
def cmd_looks(a, looks):
    for n in sorted(looks):
        print(f"{n:16s} {looks[n].get('description', '')}")


def cmd_analyze(a, looks):
    res = {}
    for f in collect(a.inputs):
        img, _ = load_image(f)
        res[f] = Scene(downsize(img, PROXY)).stats()
    print(json.dumps(res, indent=2))


def cmd_grade(a, looks):
    outdir = a.outdir
    for f in collect(a.inputs):
        t0 = time.time()
        img, meta = load_image(f)
        scene = Scene(img, use_ml=not a.no_ml)
        scene.disabled |= set(filter(None, (a.no_mask or "").split(",")))
        name = a.look
        if name == "auto":
            name = scene.stats()["suggested_looks"][0]
            log(f"[pgrade] {os.path.basename(f)}: scene={scene.stats()['scene']} -> look={name}")
        look = get_look(name, looks)
        x = grade(img, look, scene, a.strength, a.pop, seed=a.seed)
        stem = os.path.splitext(os.path.basename(f))[0]
        tag = os.path.splitext(os.path.basename(name))[0]
        ext = meta["ext"] if meta["ext"] in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp") else ".jpg"
        if a.format:
            ext = "." + a.format.lstrip(".")
        od = outdir or os.path.dirname(os.path.abspath(f))
        os.makedirs(od, exist_ok=True)
        dst = os.path.join(od, f"{stem}_{tag}{ext}")
        save_image(dst, x, meta, a.quality)
        print(f"{dst}\t{tag}\t{time.time() - t0:.1f}s")
        if a.compare:
            cdst = os.path.join(od, f"{stem}_{tag}_compare.jpg")
            compare_image(downsize(img, 2000), downsize(x, 2000), tag).save(cdst, quality=90)
            print(cdst)


def cmd_sheet(a, looks):
    names = sorted(looks) if a.looks in (None, "all") else [n.strip() for n in a.looks.split(",")]
    for f in collect(a.inputs):
        img, _ = load_image(f)
        small = downsize(img, a.size)
        scene = Scene(small, use_ml=not a.no_ml)
        scene._stats = Scene(downsize(img, PROXY), use_ml=not a.no_ml).stats()  # same call as full-size grading
        st = scene.stats()
        if a.looks == "suggested":
            names = st["suggested_looks"]
        tiles = [label(to_pil(small), "original")]
        for n in names:
            tiles.append(label(to_pil(grade(small, get_look(n, looks), scene, a.strength)), n))
        w, h = tiles[0].size
        cols = a.cols or (min(4, len(tiles)) if h >= w else (2 if len(tiles) <= 4 else 3))
        rows = math.ceil(len(tiles) / cols)
        sheet = Image.new("RGB", (cols * w + (cols - 1) * 6, rows * h + (rows - 1) * 6), "black")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * (w + 6), (i // cols) * (h + 6)))
        stem = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(a.outdir or os.path.dirname(os.path.abspath(f)), f"{stem}_sheet.jpg")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sheet.save(dst, quality=88)
        print(f"{dst}\tscene={st['scene']}\tsuggested={','.join(st['suggested_looks'])}")


def cmd_masks(a, looks):
    cols = {"sky": (60, 140, 255), "person": (40, 220, 90), "skin": (255, 150, 40), "subject": (240, 60, 200)}
    for f in collect(a.inputs):
        img, _ = load_image(f)
        scene = Scene(downsize(img, PROXY))
        names = [n for n in cols if n != "subject" or a.subject]
        tiles = []
        for n in names:
            m = scene.mask(n)[..., None]
            vis = scene.img * 0.45 + m * np.array(cols[n], np.float32) / 255 * 0.55
            t = label(to_pil(vis), f"{n} {m.mean():.0%}")
            if n == "skin":
                d = ImageDraw.Draw(t)
                for x, y, w, h, s in scene.faces():
                    d.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=2)
            tiles.append(t)
        w, h = tiles[0].size
        c = Image.new("RGB", (w * 2 + 6, h * math.ceil(len(tiles) / 2) + 6), "black")
        for i, t in enumerate(tiles):
            c.paste(t, ((i % 2) * (w + 6), (i // 2) * (h + 6)))
        stem = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(a.outdir or os.path.dirname(os.path.abspath(f)), f"{stem}_masks.jpg")
        c.save(dst, quality=85)
        print(dst)


def cmd_lut(a, looks):
    n = a.size
    idx = np.linspace(0, 1, n, dtype=np.float32)
    b, g, r = np.meshgrid(idx, idx, idx, indexing="ij")
    grid = np.stack([r.ravel(), g.ravel(), b.ravel()], -1).reshape(n * n, n, 3)
    out = grade(grid, get_look(a.look, looks), None, a.strength, pointwise=True).reshape(-1, 3)
    with open(a.out, "w") as fh:
        fh.write(f'TITLE "{a.look}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        fh.writelines(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in out)
    print(a.out, "(global colour only: masks, clarity, dehaze, bloom, grain and vignette are not in a LUT)")


def main():
    ap = argparse.ArgumentParser(description="one-step photo grading with AI masks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("looks")
    s = sub.add_parser("analyze")
    s.add_argument("inputs", nargs="+")
    s = sub.add_parser("grade")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--look", "-l", default="auto", help="look name, path to a look .json, or auto")
    s.add_argument("--strength", "-s", type=float, default=1.0)
    s.add_argument("--pop", type=float, default=0.0, help="subject pop 0..2 (needs the 170 MB subject model)")
    s.add_argument("--no-mask", help="comma list of masks to switch off, e.g. sky,skin")
    s.add_argument("--no-ml", action="store_true", help="skip all AI masks")
    s.add_argument("--compare", action="store_true")
    s.add_argument("--format", help="output format: jpg, png, tif (16-bit if source is 16-bit/RAW)")
    s.add_argument("--quality", type=int, default=95)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("-o", "--outdir")
    s = sub.add_parser("sheet")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--looks", default="all", help="all | suggested | comma list")
    s.add_argument("--strength", "-s", type=float, default=1.0)
    s.add_argument("--size", type=int, default=900)
    s.add_argument("--cols", type=int)
    s.add_argument("--no-ml", action="store_true")
    s.add_argument("-o", "--outdir")
    s = sub.add_parser("masks")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--subject", action="store_true", help="also run the subject model")
    s.add_argument("-o", "--outdir")
    s = sub.add_parser("lut")
    s.add_argument("--look", required=True)
    s.add_argument("out")
    s.add_argument("--size", type=int, default=33)
    s.add_argument("--strength", type=float, default=1.0)
    a = ap.parse_args()
    looks = load_looks()
    {"looks": cmd_looks, "analyze": cmd_analyze, "grade": cmd_grade, "sheet": cmd_sheet,
     "masks": cmd_masks, "lut": cmd_lut}[a.cmd](a, looks)


if __name__ == "__main__":
    main()
