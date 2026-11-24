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


# ============================================================== 3D LUTs
def read_cube(path):
    n, rows = None, []
    with open(path) as fh:
        for line in fh:
            t = line.strip().split()
            if not t or t[0].startswith("#"):
                continue
            if t[0] == "LUT_3D_SIZE":
                n = int(t[1])
            elif len(t) == 3 and t[0][0] in "0123456789.-":
                rows.append([float(v) for v in t])
    return np.array(rows, np.float32).reshape(n, n, n, 3)  # indexed [b, g, r]


def write_cube(path, lut, title="pgrade"):
    n = lut.shape[0]
    with open(path, "w") as fh:
        fh.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        fh.writelines(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in lut.reshape(-1, 3))


def apply_lut3d(x, lut, rows=256):
    n = lut.shape[0]
    out = np.empty_like(x)
    for r0 in range(0, x.shape[0], rows):
        c = np.clip(x[r0:r0 + rows], 0, 1) * (n - 1)
        i = np.minimum(c.astype(np.int32), n - 2)
        t = c - i
        acc = 0
        for db in (0, 1):
            wb = t[..., 2] if db else 1 - t[..., 2]
            for dg in (0, 1):
                wg = t[..., 1] if dg else 1 - t[..., 1]
                for dr in (0, 1):
                    wr = t[..., 0] if dr else 1 - t[..., 0]
                    acc = acc + (wb * wg * wr)[..., None] * lut[i[..., 2] + db, i[..., 1] + dg, i[..., 0] + dr]
        out[r0:r0 + rows] = acc
    return out


_LUT_CACHE = {}


def st_lut(x, P):
    path = P.get("lut")
    if not path:
        return x
    if path not in _LUT_CACHE:
        _LUT_CACHE[path] = read_cube(path)
    y = apply_lut3d(x, _LUT_CACHE[path])
    a = P.get("lut_amount", 1.0)
    return np.clip(x + a * (y - x), 0, 1)


def _blur3(v, sigma):
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for ax in range(3):
        pad = [(r, r) if a == ax else (0, 0) for a in range(v.ndim)]
        vp = np.pad(v, pad, mode="edge")
        v = sum(k[j] * np.take(vp, range(j, j + v.shape[ax]), axis=ax) for j in range(len(k)))
    return v


def fit_lut(before, after, n=33):
    """Learn a smooth 3D LUT mapping before -> after (same framing). Colours the photo never
    contains are filled from progressively wider neighbourhoods, so the LUT stays smooth."""
    x, d = before.reshape(-1, 3), (after - before).reshape(-1, 3)
    f = np.clip(x, 0, 1) * (n - 1)
    i0 = np.minimum(f.astype(np.int32), n - 2)
    t = f - i0
    W = np.zeros(n ** 3)
    A = np.zeros((n ** 3, 3))
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = ((t[:, 2] if db else 1 - t[:, 2]) * (t[:, 1] if dg else 1 - t[:, 1]) * (t[:, 0] if dr else 1 - t[:, 0]))
                idx = (i0[:, 0] + dr) + n * (i0[:, 1] + dg) + n * n * (i0[:, 2] + db)
                W += np.bincount(idx, w, n ** 3)
                for c in range(3):
                    A[:, c] += np.bincount(idx, w * d[:, c], n ** 3)
    W, A = W.reshape(n, n, n), A.reshape(n, n, n, 3)
    res, done = np.zeros_like(A), np.zeros((n, n, n), bool)
    for sigma in (0.8, 1.6, 3.2, 6.4, 12.8, 40.0):
        Ws, As = _blur3(W, sigma), _blur3(A, sigma)
        est = As / np.maximum(Ws, 1e-9)[..., None]
        take = ~done & (Ws > 2.0 if sigma < 40 else ~done)
        res[take] = est[take]
        done |= take
    res = _blur3(res, 0.7)
    g = np.linspace(0, 1, n, dtype=np.float32)
    bb, gg, rr = np.meshgrid(g, g, g, indexing="ij")
    return np.clip(np.stack([rr, gg, bb], -1) + res, 0, 1).astype(np.float32)


# ===================================================== matching a reference
def _wquantile(v, w, qs):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return np.interp(qs * c[-1], c, v[o])


def _zone(L, c, sig=0.14):
    return np.exp(-0.5 * ((L - c) / sig) ** 2)


def build_transfer(lt, wt, lr, wr, tone=0.6, color=0.85, cap=0.07):
    """Colour-grade transfer in Oklab from target samples to reference samples (any content).
    Tone: weighted quantile mapping of lightness, slope-limited so it never posterises.
    Colour: per-lightness-zone mean (cast) and spread (saturation) matching."""
    qs = np.linspace(0.01, 0.99, 50)
    qt, qr = _wquantile(lt[:, 0], wt, qs), _wquantile(lr[:, 0], wr, qs)
    xs = np.linspace(0, 1, 256)
    y = np.interp(xs, np.r_[0, qt, 1], np.r_[max(0, qr[0] - qt[0]), qr, min(1, qr[-1] + 1 - qt[-1])])
    d = np.clip(np.diff(y) / np.diff(xs), 0.35, 2.5)
    y = np.r_[y[0], y[0] + np.cumsum(d * np.diff(xs))]
    y = np.convolve(np.pad(y, 4, mode="edge"), np.ones(9) / 9, "valid")
    y = np.clip(xs + tone * (y - xs), 0, 1)
    Lt = np.interp(lt[:, 0], xs, y)
    centers = np.array([0.12, 0.32, 0.52, 0.72, 0.9])

    def stats(L, ab, w):
        out = []
        tot = w.sum()
        g = (w[:, None] * ab).sum(0) / tot
        gs = np.sqrt((w * ((ab - g) ** 2).sum(-1)).sum() / tot)
        for c in centers:
            z = w * _zone(L, c)
            sz = z.sum()
            conf = min(1.0, sz / tot / 0.08)  # zones the photo barely uses fall back to global stats
            m = (z[:, None] * ab).sum(0) / max(sz, 1e-9)
            sd = np.sqrt((z * ((ab - m) ** 2).sum(-1)).sum() / max(sz, 1e-9))
            out.append((conf * m + (1 - conf) * g, conf * sd + (1 - conf) * gs))
        return out
    st, sr = stats(Lt, lt[:, 1:], wt), stats(lr[:, 0], lr[:, 1:], wr)
    mt = np.array([a[0] for a in st])
    mr = np.array([a[0] for a in sr])
    shift = mr - mt
    n = np.linalg.norm(shift, axis=1, keepdims=True)
    mr = mt + shift * np.minimum(1, cap / np.maximum(n, 1e-9))  # cap the cast so nothing turns neon
    ratio = np.clip(np.array([b[1] for b in sr]) / np.maximum([a[1] for a in st], 1e-4), 0.6, 1.7)

    def apply(lab):
        L = np.interp(lab[..., 0], xs, y)
        f = lambda v: np.interp(L, centers, v)
        m_t = np.stack([f(mt[:, 0]), f(mt[:, 1])], -1)
        m_r = np.stack([f(mr[:, 0]), f(mr[:, 1])], -1)
        ab = lab[..., 1:]
        ab2 = m_r + (ab - m_t) * f(ratio)[..., None]
        out = np.empty_like(lab)
        out[..., 0] = L
        out[..., 1:] = ab + color * (ab2 - ab)
        return out.astype(np.float32)
    return apply


def bake_lut(fn, n=33):
    g = np.linspace(0, 1, n, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    grid = np.stack([r, gg, b], -1).reshape(-1, 3)
    return oklab_to_srgb(fn(to_oklab(grid))).reshape(n, n, n, 3)


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


def st_protect_skin(x, ref, m, k, orig=None):
    """Keep graded tone, but pull skin hue/chroma back toward natural."""
    if m is None or m.max() < 0.05:
        return x
    ys, xs = np.where(m > 0.02)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lx, lr = to_oklab(x[y0:y1, x0:x1]), to_oklab(ref[y0:y1, x0:x1])
    lo = lr if orig is None else to_oklab(orig[y0:y1, x0:x1])
    w = (k * m[y0:y1, x0:x1])
    hx, hr = np.degrees(np.arctan2(lx[..., 2], lx[..., 1])), np.degrees(np.arctan2(lr[..., 2], lr[..., 1]))
    Cx, Cr = np.hypot(lx[..., 1], lx[..., 2]), np.hypot(lr[..., 1], lr[..., 2])
    h = np.radians(hx + w * angdiff(hr, hx))
    C = Cx + w * (np.clip(Cx, Cr * 0.85, Cr * 1.2) - Cx)
    lx[..., 1], lx[..., 2] = C * np.cos(h), C * np.sin(h)
    # keep the person's real skin tone: a look may light, shade or tint a face, but only so far
    mm = m[y0:y1, x0:x1]
    dL = lx[..., 0] - lo[..., 0]  # measured against the original photo, auto-exposure included
    lx[..., 0] = lo[..., 0] + dL + mm * (np.clip(dL, -0.07, 0.06) - dL)
    Co, Cn = np.hypot(lo[..., 1], lo[..., 2]), np.hypot(lx[..., 1], lx[..., 2])
    f = 1 + mm * (np.clip(Cn, Co * 0.85, Co * 1.25) / np.maximum(Cn, 1e-6) - 1)  # no ashy or orange skin
    lx[..., 1] *= f
    lx[..., 2] *= f
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

    def _facezone(self, sx, sy=None, dy=0.6):
        sy = sx if sy is None else sy
        m = np.zeros((self.ph, self.pw), np.float32)
        for x, y, w, h, _ in self.faces():
            cv2.ellipse(m, (int(x + w / 2), int(y + h * dy)), (int(w * 0.5 * sx), int(h * 0.62 * sy)), 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(m, (0, 0), max(2, min(self.pw, self.ph) / 150)) if m.any() else m

    def _skin(self):
        """Skin = colour model sampled from the detected faces (lightness + colour, so brown hair and
        warm fabrics are rejected), restricted to the people in the photo."""
        faces = self.faces()
        if not faces:
            return np.zeros((self.ph, self.pw), np.float32)
        lab = to_oklab(self.px)
        samp = np.zeros((self.ph, self.pw), np.uint8)
        for x, y, w, h, _ in faces:
            cv2.ellipse(samp, (int(x + w / 2), int(y + h * 0.6)), (max(1, int(w * 0.26)), max(1, int(h * 0.26))), 0, 0, 360, 1, -1)
        sel = (samp > 0) & (lab[..., 0] > 0.3) & (lab[..., 0] < 0.97)
        if sel.sum() < 20:
            return self.proxy_mask("face")
        v = lab[sel]
        mu = np.median(v, 0)
        cov = np.cov(v.T) + np.diag([4e-4, 2e-5, 2e-5])
        cov[0, 0] *= 4.0  # allow light and shadow on the same skin
        d = lab - mu
        d2 = np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d)
        p = np.clip(1 - (np.sqrt(d2) - 2.2) / 1.8, 0, 1)
        L = lab[..., 0]
        r = max(1, round(min(self.pw, self.ph) / 400))
        sd = np.sqrt(np.maximum(box(L * L, r) - box(L, r) ** 2, 0))
        p *= 1 - smoothstep(0.03, 0.07, sd)
        person = cv2.dilate(self.proxy_mask("person"), np.ones((5, 5), np.uint8))
        gate = np.maximum(person, self._facezone(1.0)) if person.mean() > 0.005 else self._facezone(1.6)
        return self._refine(p * np.clip(gate, 0, 1), r_div=150, lo=0.25, hi=0.75)

    def _faceskin(self):
        """Skin on and right around faces only: what retouching should touch."""
        zone = np.clip(self._facezone(1.12, 1.3, 0.68) * 1.5, 0, 1)  # face width, extended down to the neck
        return self.proxy_mask("skin") * zone

    def _subject(self):
        x = cv2.resize(self.pu8, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (x / max(x.max(), 1) - 0.5).transpose(2, 0, 1)[None]
        s = self._ort("subject")
        o = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
        o = (o - o.min()) / (o.max() - o.min() + 1e-6)
        return self._refine(o, r_div=120, lo=0.35, hi=0.65)

    # ---- mask access -----------------------------------------------------
    NAMES = ("sky", "person", "face", "skin", "faceskin", "subject")

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
                m = (1 - smoothstep(wd, wd + 15, np.abs(angdiff(hh, c)))) * smoothstep(0.035, 0.08, np.hypot(lab[..., 1], lab[..., 2]))
            elif name == "grad":
                side, _, ext = arg.partition(":")
                ext = float(ext or 0.5)
                yy, xx = np.mgrid[:self.H, :self.W].astype(np.float32)
                t = {"top": yy / self.H, "bottom": 1 - yy / self.H, "left": xx / self.W, "right": 1 - xx / self.W}[side or "top"]
                m = 1 - smoothstep(0, ext, t)
            elif name == "radial":  # radial:SIZE or radial:CX,CY,SIZE (fractions of the frame)
                v = [float(t) for t in arg.split(",")] if arg else [0.6]
                cx, cy, size = (0.5, 0.5, v[0]) if len(v) == 1 else v[:3]
                yy, xx = np.ogrid[:self.H, :self.W]
                sc = min(self.W, self.H) / 2
                d = np.sqrt(((xx - cx * self.W) / sc) ** 2 + ((yy - cy * self.H) / sc) ** 2)
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
        neutral = (C < 0.07) & (L > 0.45) & (sky < 0.5)  # warm LIGHT shows up on whites and greys
        s["warm_neutrals"] = float(lab[..., 2][neutral].mean()) if neutral.mean() > 0.02 else 0.0
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
    elif s["face_area"] > 0.003:
        scene = "people"
    elif sky and sky["chroma"] < 0.035 and sky["lightness"] > 0.55 and s["sky_fraction"] > 0.08 \
            and s["warm_highlights"] < 0.035 and s["mean_chroma"] < 0.035:  # grey sky AND a grey scene
        scene = "overcast"
    elif blue_sky or (sky and s["ground_luma"] > 0.45 and s["mean_chroma"] >= 0.035):  # incl. hazy-white sky on a sunny day
        scene = "daylight"
    elif False:
        scene = "daylight"
    elif s["warm_highlights"] > 0.05 and s.get("warm_neutrals", 0) > 0.02:
        scene = "golden_hour"
    else:
        scene = "other"
    looks = {"portrait": ["jp-airy", "film-warm", "jp-film"],
             "golden_hour": ["gta6", "golden", "teal-orange"],
             "blue_hour": ["vice-night", "neon-noir", "teal-orange"],
             "night_neon": ["neon-noir", "vice-night", "gta6"],
             "night": ["vice-night", "neon-noir", "moody"],
             "overcast": ["moody", "classic-chrome", "bw"],
             "daylight": ["jp-summer", "teal-orange", "natural"],
             "people": ["jp-film", "natural", "film-warm"],
             "other": ["natural", "jp-film", "classic-chrome"]}[scene]
    return scene, looks


# ================================================================ looks
def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k == "local" and isinstance(out.get(k), list) and not over.get("replace_local"):
            out[k] = out[k] + copy.deepcopy(v)  # a recipe ADDS its local moves to the base look's
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_luts(j, folder):
    fix = lambda p: p if (not p or os.path.isabs(p)) else os.path.join(folder, p)
    if j.get("lut"):
        j["lut"] = fix(j["lut"])
    for loc in j.get("local", []):
        if loc.get("adjust", {}).get("lut"):
            loc["adjust"]["lut"] = fix(loc["adjust"]["lut"])
    return j


def load_looks():
    raw = {}
    for d in LOOK_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(f) as fh:
                j = json.load(fh)
            _resolve_luts(j, os.path.dirname(os.path.abspath(f)))
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
        _resolve_luts(j, os.path.dirname(os.path.abspath(name)))
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
        f = 0.15 if keep else (0.3 if s["scene"] == "portrait" else 0.6)  # flattering skin beats neutral grey
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
    x = st_lut(x, P)
    x = st_local_contrast(x, P.get("clarity", 0), P.get("texture", 0))
    return st_color(x, P)


def grade(img, look, scene, strength=1.0, pop=0.0, pointwise=False, seed=7, retouch=0.0):
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
        if retouch:
            locs += [{"mask": "faceskin", "adjust": {"texture": -30 * retouch, "clarity": -12 * retouch, "exposure": 0.06 * retouch}}]
        if pop and not (0.004 < scene.proxy_mask("subject").mean() < 0.75):
            log("[pgrade] no clear subject found -> --pop skipped")
            pop = 0
        if pop:
            locs += [{"mask": "subject", "adjust": {"exposure": 0.12 * pop, "clarity": 12 * pop, "vibrance": 8 * pop}},
                     {"mask": "background", "adjust": {"exposure": -0.12 * pop, "saturation": -8 * pop}}]
        scene.used_locals = locs
        for loc in locs:
            m = scene.expr(loc["mask"]) * loc.get("amount", 1.0)
            if m.max() < 0.02:
                continue
            y = develop(x, loc["adjust"])
            x = x + m[..., None] * (y - x)
        if scene.faces():
            x = st_protect_skin(x, ref, scene.mask("skin"), P.get("protect_skin", 0.0), orig=img)
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
    icc = im.info.get("icc_profile")
    if icc and im.mode in ("RGB", "RGBA"):
        im = to_srgb(im, icc)  # grade in sRGB so every look behaves the same on every camera
    meta["icc"] = SRGB_ICC
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


def _srgb_icc():
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


SRGB_ICC = _srgb_icc()


def to_srgb(im, icc):
    """Convert an image with an embedded profile (e.g. iPhone Display P3) to sRGB."""
    import io
    from PIL import ImageCms
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if "srgb" in (ImageCms.getProfileDescription(src) or "").lower():
            return im
        alpha = im.getchannel("A") if im.mode == "RGBA" else None
        out = ImageCms.profileToProfile(im.convert("RGB"), src, ImageCms.createProfile("sRGB"),
                                        renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB")
        if alpha is not None:
            out.putalpha(alpha)
        return out
    except Exception as e:
        log(f"[pgrade] colour profile not converted ({e}); treating as sRGB")
        return im


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
        x = grade(img, look, scene, a.strength, a.pop, seed=a.seed, retouch=a.retouch)
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
        if a.show_masks:
            mp = masks_used_preview(scene, getattr(scene, "used_locals", []), os.path.join(od, f"{stem}_{tag}_masks.jpg"))
            print(mp or "(no masked adjustments in this grade)")


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
    cols = {"sky": (60, 140, 255), "person": (40, 220, 90), "skin": (255, 150, 40), "faceskin": (255, 230, 60),
            "subject": (240, 60, 200)}
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
    out = grade(grid, get_look(a.look, looks), None, a.strength, pointwise=True)
    write_cube(a.out, out.reshape(n, n, n, 3), a.look)
    print(a.out, "(global colour only: masks, clarity, dehaze, bloom, grain and vignette are not in a LUT)")


def cmd_match(a, looks):
    """Copy the colour style of a reference image (any content) onto the user's photo."""
    t, _ = load_image(a.photo)
    r, _ = load_image(a.reference)
    ts, rs = downsize(t, 640), downsize(r, 640)
    st, sr = Scene(ts), Scene(rs)
    k = a.strength
    lt, lr = to_oklab(ts).reshape(-1, 3), to_oklab(rs).reshape(-1, 3)
    skt, skr = st.proxy_mask("sky").ravel(), sr.proxy_mask("sky").ravel()
    # skin never drives the palette (and is protected when grading)
    wt = (1 - st.proxy_mask("skin").ravel()) * (1 - skt if skt.mean() > 0.05 else 1)
    wr = (1 - sr.proxy_mask("skin").ravel()) * (1 - skr if skr.mean() > 0.05 else 1)
    g = build_transfer(lt, wt + 1e-6, lr, wr + 1e-6, tone=0.6 * k, color=0.85 * k)
    od = a.outdir or os.path.expanduser("~/.config/photo-grade/looks")
    os.makedirs(os.path.join(od, "luts"), exist_ok=True)
    write_cube(os.path.join(od, "luts", f"{a.name}.cube"), bake_lut(g), a.name)
    look = {"name": a.name, "description": a.description or f"Colour matched to {os.path.basename(a.reference)}.",
            "best_for": "the photo it was matched for, and similar scenes", "matched_to": os.path.basename(a.reference),
            "auto": {"exposure": False, "wb": False}, "lut": f"luts/{a.name}.cube", "protect_skin": 0.6, "sharpness": 15}
    if skt.mean() > 0.05 and skr.mean() > 0.05:  # sky follows the reference's sky, not its ground
        gt = g(lt)
        sky = build_transfer(gt, skt + 1e-6, lr, skr + 1e-6, tone=0.5 * k, color=0.85 * k, cap=0.1)
        write_cube(os.path.join(od, "luts", f"{a.name}_sky.cube"), bake_lut(sky), a.name + " sky")
        look["local"] = [{"mask": "sky", "adjust": {"lut": f"luts/{a.name}_sky.cube"}}]
    with open(os.path.join(od, f"{a.name}.json"), "w") as fh:
        json.dump(look, fh, indent=2)
    looks = load_looks() if od in LOOK_DIRS else looks
    lk = get_look(os.path.join(od, f"{a.name}.json"), looks)
    out = grade(ts, lk, st)
    h = ts.shape[0]
    rr = cv2.resize(rs, (round(rs.shape[1] * h / rs.shape[0]), h), interpolation=cv2.INTER_AREA)
    tiles = [label(to_pil(ts), "your photo"), label(to_pil(out), "matched"), label(to_pil(rr), "reference")]
    c = Image.new("RGB", (sum(tl.width for tl in tiles) + 12, h), "black")
    xo = 0
    for tl in tiles:
        c.paste(tl, (xo, 0))
        xo += tl.width + 6
    cpath = os.path.join(od, f"{a.name}_match_check.jpg")
    c.save(cpath, quality=88)
    print(f"{os.path.join(od, a.name + '.json')}\n{cpath}")


def masks_used_preview(scene, locs, path):
    tiles = []
    for loc in locs:
        m = downsize(scene.expr(loc["mask"]), 700)
        base = downsize(scene.img, 700)
        vis = base * 0.35 + m[..., None] * np.array([1.0, 0.55, 0.15], np.float32) * 0.65 + (m[..., None] * base * 0.35)
        tiles.append(label(to_pil(vis), f'{loc["mask"]}  {", ".join(f"{k} {v}" for k, v in loc["adjust"].items() if k != "lut")}'[:60], 18))
    if not tiles:
        return None
    w, h = tiles[0].size
    cols = 2 if len(tiles) > 1 else 1
    c = Image.new("RGB", (cols * w + 6 * (cols - 1), math.ceil(len(tiles) / cols) * (h + 6)), "black")
    for i, tl in enumerate(tiles):
        c.paste(tl, ((i % cols) * (w + 6), (i // cols) * (h + 6)))
    c.save(path, quality=85)
    return path


# ===================================================================== scopes
SKIN_HUE, SKIN_LO, SKIN_HI = 47.0, 32.0, 63.0  # Oklab hue of real skin (measured across skin tones)


def _cast(a, b):
    parts = []
    if abs(b) > 0.006:
        parts.append(f"{'warm' if b > 0 else 'cool'} {b:+.3f}")
    if abs(a) > 0.006:
        parts.append(f"{'magenta' if a > 0 else 'green'} {a:+.3f}")
    return ", ".join(parts) or "neutral"


def region_stats(x, w):
    sel = w > 0.5
    if sel.mean() < 0.004:
        return None
    px = x[sel]
    L = px @ LUMA
    lab = to_oklab(px)
    C = np.hypot(lab[:, 1], lab[:, 2])
    h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    st = {"area": float(sel.mean()), "luma_p1_50_99": [round(float(v), 3) for v in np.percentile(L, [1, 50, 99])],
          "clipped_highlights": float((px.max(-1) >= 0.995).mean()), "crushed_shadows": float((px.max(-1) <= 0.012).mean()),
          "colour_clipping": float(((px.max(-1) >= 0.995) | (px.min(-1) <= 0.003)).__and__(C > 0.08).mean()),
          "chroma_mean": float(C.mean()), "chroma_p95": float(np.percentile(C, 95)),
          "hue_median": float(np.degrees(np.arctan2(np.median(lab[:, 2]), np.median(lab[:, 1]))) % 360),
          "lightness_median": float(np.median(lab[:, 0])), "chroma_median": float(np.median(C))}
    neu = (C < 0.04) & (lab[:, 0] > 0.25) & (lab[:, 0] < 0.95)
    st["neutral_cast"] = _cast(*lab[neu][:, 1:].mean(0)) if neu.mean() > 0.02 else "n/a"
    W = hsl_weights(h) * C[:, None]
    share = W.sum(0) / max(W.sum(), 1e-9)
    st["main_hues"] = ", ".join(f"{HSL_BANDS[i][0]} {share[i]:.0%}" for i in np.argsort(-share)[:3] if share[i] > 0.1)
    return st


def _fmt(name, st):
    p = st["luma_p1_50_99"]
    s = (f"{name:8s} {st['area']:5.0%}  luma p1/50/99 {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}  clip hi {st['clipped_highlights']:.1%}"
         f" lo {st['crushed_shadows']:.1%}  sat {st['chroma_mean']:.3f} (p95 {st['chroma_p95']:.3f})  cast {st['neutral_cast']}")
    if st["main_hues"]:
        s += f"  hues: {st['main_hues']}"
    if name == "skin":
        s += f"  | skin hue {st['hue_median']:.0f}° (line {SKIN_HUE:.0f}°) L {st['lightness_median']:.2f} C {st['chroma_median']:.3f}"
    if name == "sky":
        s += f"  | sky hue {st['hue_median']:.0f}° L {st['lightness_median']:.2f} C {st['chroma_median']:.3f}"
    return s


def _warnings(regions, before=None):
    w = []
    a = regions.get("all")
    if a and a["clipped_highlights"] > 0.01:
        w.append(f"highlights clipped on {a['clipped_highlights']:.1%} of the frame")
    if a and a["crushed_shadows"] > 0.03:
        w.append(f"shadows crushed on {a['crushed_shadows']:.1%} of the frame")
    if a and a["colour_clipping"] > 0.01:
        w.append(f"saturated colours hit the gamut edge on {a['colour_clipping']:.1%} (flat, posterised patches)")
    sk = regions.get("skin")
    if sk:
        if sk["hue_median"] > SKIN_HI:
            w.append(f"skin too yellow/orange (hue {sk['hue_median']:.0f}°, real skin {SKIN_LO:.0f}–{SKIN_HI:.0f}°)")
        elif sk["hue_median"] < SKIN_LO:
            w.append(f"skin too pink/magenta (hue {sk['hue_median']:.0f}°, real skin {SKIN_LO:.0f}–{SKIN_HI:.0f}°)")
        if sk["chroma_median"] < 0.022:
            w.append(f"skin looks grey/ashy (chroma {sk['chroma_median']:.3f})")
        if sk["chroma_median"] > 0.13:
            w.append(f"skin oversaturated (chroma {sk['chroma_median']:.3f})")
    sy = regions.get("sky")
    if sy and sk and sy["area"] > 0.3 and sy["chroma_median"] < 0.04 and not (180 < sy["hue_median"] < 300):
        w.append("the 'sky' mask may be a plain wall or backdrop behind the person; check `masks` before using sky moves")
    if sy and sy["clipped_highlights"] > 0.03:
        w.append(f"sky blown on {sy['clipped_highlights']:.0%} of the sky")
    if before:
        b0, a0 = before.get("skin"), regions.get("skin")
        if b0 and a0:
            dh = angdiff(a0["hue_median"], b0["hue_median"])
            if abs(dh) > 8:
                w.append(f"skin hue moved {dh:+.0f}° from the original")
            if abs(a0["lightness_median"] - b0["lightness_median"]) > 0.07:
                w.append(f"skin lightness moved {a0['lightness_median'] - b0['lightness_median']:+.2f} from the original")
        if a and before.get("all") and a["clipped_highlights"] - before["all"]["clipped_highlights"] > 0.005:
            w.append(f"the grade added {a['clipped_highlights'] - before['all']['clipped_highlights']:.1%} clipped highlights")
    return w


def scope_panels(x, title):
    L = luma(x)
    H, W = L.shape
    xs = cv2.resize(L, (512, max(1, round(H * 512 / W))), interpolation=cv2.INTER_AREA)
    wf = np.zeros((200, 512), np.float32)
    rows = np.clip((1 - xs) * 199, 0, 199).astype(int)
    for c in range(512):
        np.add.at(wf[:, c], rows[:, c], 1)
    wf = np.log1p(wf) / max(np.log1p(wf).max(), 1e-6)
    wfi = np.stack([wf * 0.6, wf, wf * 0.6], -1)
    wfi[0, :] = wfi[199, :] = [0.8, 0.2, 0.2]  # 0% and 100% lines
    wfi[100, ::4] = 0.35
    lab = to_oklab(cv2.resize(x, (512, max(1, round(H * 512 / W))), interpolation=cv2.INTER_AREA)).reshape(-1, 3)
    n, R = 200, 0.22
    ij = np.clip(((lab[:, 1:] / R) * (n / 2) + n / 2).astype(int), 0, n - 1)
    vs = np.zeros((n, n), np.float32)
    np.add.at(vs, (n - 1 - ij[:, 1], ij[:, 0]), 1)
    vs = np.log1p(vs) / max(np.log1p(vs).max(), 1e-6)
    vsi = np.repeat(vs[..., None], 3, -1) * np.array([0.9, 0.95, 1.0], np.float32)
    vimg = to_pil(vsi)
    d = ImageDraw.Draw(vimg)
    for rr in (0.05, 0.1, 0.15, 0.2):
        r = rr / R * n / 2
        d.ellipse([n / 2 - r, n / 2 - r, n / 2 + r, n / 2 + r], outline=(60, 60, 60))
    ang = math.radians(SKIN_HUE)
    d.line([n / 2, n / 2, n / 2 + math.cos(ang) * n / 2, n / 2 - math.sin(ang) * n / 2], fill=(255, 170, 90), width=1)
    for name, hh in (("R", 25), ("Y", 100), ("G", 140), ("C", 195), ("B", 255), ("M", 335)):
        a2 = math.radians(hh)
        d.text((n / 2 + math.cos(a2) * 0.19 / R * n / 2 - 4, n / 2 - math.sin(a2) * 0.19 / R * n / 2 - 6), name, fill=(200, 200, 200), font=_font(11))
    hist = Image.new("RGB", (256, 200), "black")
    hd = ImageDraw.Draw(hist)
    q = (np.clip(x, 0, 1) * 255).astype(np.uint8).reshape(-1, 3)
    hs = [np.bincount(q[:, c], minlength=256) for c in range(3)]
    top = max(h[2:-2].max() for h in hs) or 1
    for c, col in enumerate([(255, 70, 70), (70, 255, 70), (90, 140, 255)]):
        pts = [(i, 199 - min(199, int(hs[c][i] / top * 190))) for i in range(256)]
        hd.line(pts, fill=col, width=1)
    panel = Image.new("RGB", (512 + 200 + 256 + 12, 222), (20, 20, 20))
    panel.paste(to_pil(wfi), (0, 22))
    panel.paste(vimg, (518, 22))
    panel.paste(hist, (724, 22))
    ImageDraw.Draw(panel).text((4, 3), f"{title}: waveform | vectorscope (orange line = skin) | RGB histogram", fill="white", font=_font(14))
    return panel


def _crop_boxes(scene, full_h, full_w, size=384):
    """Spots that downscaled previews hide: the biggest face, the sky edge, the brightest highlight."""
    sx, sy = full_w / scene.pw, full_h / scene.ph
    boxes = []
    if scene.faces():
        x, y, w, h, _ = max(scene.faces(), key=lambda f: f[2] * f[3])
        cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
        boxes.append(("face", cx, cy))
    sky = scene.proxy_mask("sky")
    if 0.03 < sky.mean() < 0.97:
        g = cv2.morphologyEx(sky, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
        yy, xx = np.unravel_index(np.argmax(cv2.GaussianBlur(g * luma(scene.px), (0, 0), 3)), g.shape)
        boxes.append(("sky edge", xx * sx, yy * sy))
    L = cv2.GaussianBlur(luma(scene.px), (0, 0), 4)
    yy, xx = np.unravel_index(np.argmax(L), L.shape)
    boxes.append(("highlight", xx * sx, yy * sy))
    out = []
    for name, cx, cy in boxes:
        x0 = int(np.clip(cx - size / 2, 0, max(0, full_w - size)))
        y0 = int(np.clip(cy - size / 2, 0, max(0, full_h - size)))
        out.append((name, y0, min(full_h, y0 + size), x0, min(full_w, x0 + size)))
    return out


def cmd_scopes(a, looks):
    img, _ = load_image(a.photo)
    versions = [("original", img)]
    if a.vs:
        g, _ = load_image(a.vs)
        if g.shape[:2] != img.shape[:2]:
            g = cv2.resize(g, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        versions.append(("graded", g))
    scene = Scene(downsize(img, PROXY))
    regions = {"all": np.ones((scene.ph, scene.pw), np.float32), "sky": scene.proxy_mask("sky")}
    if scene.faces():
        regions["skin"] = scene.proxy_mask("faceskin")
    if a.subject:
        regions["subject"] = scene.proxy_mask("subject")
    for e in a.mask or []:
        regions[e] = downsize(scene.expr(e), PROXY)
    report, prev = {}, None
    for name, v in versions:
        pv = cv2.resize(v, (scene.pw, scene.ph), interpolation=cv2.INTER_AREA)
        st = {k: region_stats(pv, m) for k, m in regions.items()}
        st = {k: s for k, s in st.items() if s}
        report[name] = {"regions": st, "warnings": _warnings(st, prev)}
        print(f"== {name}")
        for k, s in st.items():
            print(_fmt(k, s))
        for wmsg in report[name]["warnings"]:
            print(f"!! {wmsg}")
        prev = st
    stem = os.path.splitext(os.path.basename(a.vs or a.photo))[0]
    od = a.outdir or os.path.dirname(os.path.abspath(a.vs or a.photo))
    os.makedirs(od, exist_ok=True)
    panels = [scope_panels(cv2.resize(v, (scene.pw, scene.ph), interpolation=cv2.INTER_AREA), n) for n, v in versions]
    sc = Image.new("RGB", (panels[0].width, sum(pn.height for pn in panels)), "black")
    for i, pn in enumerate(panels):
        sc.paste(pn, (0, i * pn.height))
    sc.save(os.path.join(od, f"{stem}_scopes.jpg"), quality=90)
    boxes = _crop_boxes(scene, *img.shape[:2])
    tiles = []
    for bname, y0, y1, x0, x1 in boxes:
        row = [label(to_pil(v[y0:y1, x0:x1]), f"{bname} 100% ({n})", 16) for n, v in versions]
        tiles.append(row)
    if tiles:
        cw = max(t.width for r in tiles for t in r)
        chh = max(t.height for r in tiles for t in r)
        cr = Image.new("RGB", (len(versions) * (cw + 6), len(tiles) * (chh + 6)), "black")
        for i, r in enumerate(tiles):
            for j, t in enumerate(r):
                cr.paste(t, (j * (cw + 6), i * (chh + 6)))
        cr.save(os.path.join(od, f"{stem}_crops.jpg"), quality=92)
    print(os.path.join(od, f"{stem}_scopes.jpg"))
    print(os.path.join(od, f"{stem}_crops.jpg"))
    if a.json:
        with open(os.path.join(od, f"{stem}_scopes.json"), "w") as fh:
            json.dump(report, fh, indent=2)


def cmd_learn(a, looks):
    """Learn a look from a before/after pair of the same photo (any editor: Lightroom, VSCO...)."""
    b, _ = load_image(a.before)
    t, _ = load_image(a.after)
    ra, rb = t.shape[1] / t.shape[0], b.shape[1] / b.shape[0]
    if abs(ra - rb) > 0.01:
        if not a.loose:
            sys.exit("before and after must have the same framing (use --loose for an AI restyle)")
        H, W = b.shape[:2]  # centre-crop the original to the restyle's aspect ratio
        if ra > rb:
            nh = int(W / ra)
            b = b[(H - nh) // 2:(H - nh) // 2 + nh]
        else:
            nw = int(H * ra)
            b = b[:, (W - nw) // 2:(W - nw) // 2 + nw]
    b = downsize(b, 320 if a.loose else 1400)
    t = cv2.resize(t, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    if a.loose:  # an AI restyle moves details a little: compare colours, not edges
        b, t = cv2.GaussianBlur(b, (0, 0), 2), cv2.GaussianBlur(t, (0, 0), 2)
    lut = fit_lut(b, t, 33)
    y = apply_lut3d(b, lut)
    log(f"[pgrade] colour fit: error {np.abs(b - t).mean() * 255:.1f} -> {np.abs(y - t).mean() * 255:.1f} (of 255)")
    bs, ts = downsize(b, 900), downsize(t, 900)
    ys = apply_lut3d(bs, lut)
    err = lambda cl, tx: np.abs(st_local_contrast(ys.copy(), cl, tx) - ts).mean()
    cl, tx = 0, 0
    for _ in range(2):  # coordinate search for local contrast the LUT can't express
        cl = min(range(-20, 81, 10), key=lambda v: err(v, tx))
        tx = min(range(-20, 61, 10), key=lambda v: err(cl, v))
    tx = min(tx, 40)  # stay gentle on skin
    if a.loose:
        cl, tx = 0, 0  # detail can't be learned from a blurred, re-drawn image
    od = a.outdir or os.path.expanduser("~/.config/photo-grade/looks")
    os.makedirs(os.path.join(od, "luts"), exist_ok=True)
    write_cube(os.path.join(od, "luts", f"{a.name}.cube"), lut, a.name)
    look = {"name": a.name, "description": a.description or f"Learned from {os.path.basename(a.after)}.",
            "best_for": a.best_for or "photos similar to the example it was learned from",
            "learned_from": [os.path.basename(a.before), os.path.basename(a.after)],
            "auto": {"exposure": False, "wb": False}, "lut": f"luts/{a.name}.cube",
            "clarity": cl, "texture": tx, "protect_skin": 0.6, "sharpness": 20}
    with open(os.path.join(od, f"{a.name}.json"), "w") as fh:
        json.dump(look, fh, indent=2)
    chk = np.concatenate([b, st_local_contrast(y, cl, tx), t], 1)
    cpath = os.path.join(od, f"{a.name}_learn_check.jpg")
    label(to_pil(downsize(chk, 2400)), "before | learned | your edit").save(cpath, quality=88)
    print(f"{os.path.join(od, a.name + '.json')}\tclarity={cl} texture={tx}\n{cpath}")


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
    s.add_argument("--retouch", type=float, default=0.0, help="skin smoothing + face light 0..2 (faces only)")
    s.add_argument("--no-mask", help="comma list of masks to switch off, e.g. sky,skin")
    s.add_argument("--no-ml", action="store_true", help="skip all AI masks")
    s.add_argument("--compare", action="store_true")
    s.add_argument("--show-masks", dest="show_masks", action="store_true", help="save a preview of every mask the grade used")
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
    s = sub.add_parser("scopes", help="measure colour: per-region numbers, waveform/vectorscope, 100% crops")
    s.add_argument("photo", help="the original photo")
    s.add_argument("--vs", help="the graded version to compare against the original")
    s.add_argument("--mask", action="append", help="extra region to measure (mask expression); repeatable")
    s.add_argument("--subject", action="store_true", help="also measure the subject (needs the subject model)")
    s.add_argument("--json", action="store_true")
    s.add_argument("-o", "--outdir")
    s = sub.add_parser("match", help="copy the colour style of a reference image onto a photo")
    s.add_argument("photo")
    s.add_argument("reference")
    s.add_argument("--name", required=True)
    s.add_argument("--strength", type=float, default=1.0)
    s.add_argument("--description")
    s.add_argument("-o", "--outdir", help="default ~/.config/photo-grade/looks")
    s = sub.add_parser("learn", help="learn a look from a before/after pair of the same photo")
    s.add_argument("before")
    s.add_argument("after")
    s.add_argument("--name", required=True)
    s.add_argument("--description")
    s.add_argument("--best-for", dest="best_for")
    s.add_argument("--loose", action="store_true", help="AFTER is an AI restyle of BEFORE (details shifted / different crop)")
    s.add_argument("-o", "--outdir", help="default ~/.config/photo-grade/looks")
    s = sub.add_parser("lut")
    s.add_argument("--look", required=True)
    s.add_argument("out")
    s.add_argument("--size", type=int, default=33)
    s.add_argument("--strength", type=float, default=1.0)
    a = ap.parse_args()
    looks = load_looks()
    {"looks": cmd_looks, "analyze": cmd_analyze, "grade": cmd_grade, "sheet": cmd_sheet,
     "masks": cmd_masks, "lut": cmd_lut, "learn": cmd_learn, "match": cmd_match, "scopes": cmd_scopes}[a.cmd](a, looks)


if __name__ == "__main__":
    main()
