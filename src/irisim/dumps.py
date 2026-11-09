"""Turning GL texture contents into inspectable PNGs + statistics."""
import os
import time

import numpy as np
from PIL import Image


def _srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def stats(arr, channels=4):
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[..., None]
    a = a[..., :channels].astype(np.float64)
    nan = int(np.isnan(a).sum())
    inf = int(np.isinf(a).sum())
    f = np.where(np.isfinite(a), a, 0.0)
    return {"min": f.reshape(-1, f.shape[-1]).min(0), "max": f.reshape(-1, f.shape[-1]).max(0),
            "mean": f.reshape(-1, f.shape[-1]).mean(0), "nan": nan, "inf": inf}


def fmt_stats(s):
    def v(x):
        return "(" + ", ".join(f"{t:.4g}" for t in x) + ")"
    extra = ""
    if s["nan"] or s["inf"]:
        extra = f"  !! NaN={s['nan']} Inf={s['inf']}"
    return f"min {v(s['min'])} max {v(s['max'])} mean {v(s['mean'])}{extra}"


def save_rgb(path, img01, flip=True):
    """img01: (h, w, 3|4) floats in [0,1]."""
    a = np.clip(np.asarray(img01, np.float64), 0, 1)
    if flip:
        a = a[::-1]
    u8 = (a * 255.0 + 0.5).astype(np.uint8)
    if u8.shape[-1] == 1:
        u8 = u8[..., 0]
    Image.fromarray(np.ascontiguousarray(u8)).save(path)


def visualize(arr, kind, channels):
    """Return (rgb image in [0,1], alpha image or None) for a texture read-back.

    float  -> Reinhard tonemap x/(1+x) + sRGB (NaN magenta, +Inf cyan)
    unorm  -> as is
    snorm  -> *0.5+0.5
    int/uint -> hashed colour per value (0 = black)
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[..., None]
    if kind in ("uint", "int"):
        v = a[..., 0].astype(np.uint64)
        h = (v * np.uint64(2654435761)) & np.uint64(0xFFFFFF)
        rgb = np.stack([(h >> np.uint64(16)) & np.uint64(255), (h >> np.uint64(8)) & np.uint64(255),
                        h & np.uint64(255)], -1).astype(np.float64) / 255.0
        rgb = rgb * 0.7 + 0.3
        rgb[v == 0] = 0.0
        return rgb, None
    x = a.astype(np.float64)
    nan = np.isnan(x[..., :3]).any(-1)
    pinf = np.isposinf(x[..., :3]).any(-1)
    x = np.where(np.isfinite(x), x, 0.0)
    if channels == 1 or x.shape[-1] == 1:
        rgb = np.repeat(x[..., :1], 3, -1)
    elif channels == 2 or x.shape[-1] == 2:
        rgb = np.concatenate([x[..., :2], np.zeros_like(x[..., :1])], -1)
    else:
        rgb = x[..., :3]
    if kind == "float":
        rgb = _srgb(np.maximum(rgb, 0) / (1.0 + np.maximum(rgb, 0)))
    elif kind == "snorm":
        rgb = rgb * 0.5 + 0.5
    rgb = np.clip(rgb, 0, 1)
    rgb[nan] = (1, 0, 1)
    rgb[pinf] = (0, 1, 1)
    alpha = None
    if channels == 4 and x.shape[-1] == 4:
        al = x[..., 3]
        if np.ptp(al) > 1e-6:
            if kind == "float":
                al = al / (1.0 + np.abs(al))
            alpha = np.repeat(np.clip(al, 0, 1)[..., None], 3, -1)
    return rgb, alpha


def depth_vis(d, near, far):
    """Linearised depth -> grey (0 near, 1 at `far`), sky (1.0) white."""
    d = np.asarray(d, np.float64)
    z = d * 2.0 - 1.0
    lin = 2.0 * near * far / np.maximum(far + near - z * (far - near), 1e-9)
    g = np.clip(np.sqrt(lin / far), 0, 1)
    g[d >= 1.0] = 1.0
    return np.repeat(g[..., None], 3, -1)


class _Timer:
    def __init__(self, d):
        self.d = d

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *a):
        self.d.seconds += time.perf_counter() - self.t


class Dumper:
    def __init__(self, out_dir, log):
        self.dir = out_dir
        self.log = log
        self.seq = 0
        self.seconds = 0.0
        os.makedirs(out_dir, exist_ok=True)

    def timing(self):
        return _Timer(self)

    def path(self, name):
        return os.path.join(self.dir, name)

    def texture(self, label, tex, alpha=True):
        with self.timing():
            return self._texture(label, tex, alpha)

    def depth(self, label, tex, near, far):
        with self.timing():
            return self._depth(label, tex, near, far)

    def _texture(self, label, tex, alpha=True):
        arr = tex.read()
        kind = tex.kind
        ch = tex.info["channels"]
        st = stats(arr, ch if kind != "depth" else 1)
        fname = f"{self.seq:03d}_{label}.png"
        self.seq += 1
        if kind == "depth":
            # normalise the occupied depth range (1.0 = cleared) so small ranges stay visible
            d = np.asarray(arr, np.float64)
            m = d < 1.0
            lo, hi = (d[m].min(), d[m].max()) if m.any() else (0.0, 1.0)
            g = np.where(m, (d - lo) / max(hi - lo, 1e-9) * 0.9, 1.0)
            save_rgb(self.path(fname), np.repeat(g[..., None], 3, -1))
        else:
            rgb, al = visualize(arr, kind, ch)
            save_rgb(self.path(fname), rgb)
            if al is not None and alpha:
                save_rgb(self.path(fname.replace(".png", "_a.png")), al)
        self.log.info(f"    dump {fname:44s} {tex.info['name']:8s} {fmt_stats(st)}")
        return st

    def _depth(self, label, tex, near, far):
        arr = tex.read()
        fname = f"{self.seq:03d}_{label}.png"
        self.seq += 1
        save_rgb(self.path(fname), depth_vis(arr, near, far))
        st = stats(arr, 1)
        self.log.info(f"    dump {fname:44s} depth    {fmt_stats(st)}")
