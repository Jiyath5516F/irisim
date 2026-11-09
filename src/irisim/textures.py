"""Procedural 16x16 block textures (vanilla-like), atlas packing and optional
labPBR normal / specular atlases."""
import zlib

import numpy as np

T = 16


def _rng(name):
    return np.random.default_rng(zlib.crc32(name.encode()))


def _rgba(rgb, a=1.0):
    img = np.zeros((T, T, 4), np.float32)
    img[..., :3] = rgb
    img[..., 3] = a
    return img


def _noise(name, lo, hi, smooth=0):
    r = _rng(name).random((T, T)).astype(np.float32)
    for _ in range(smooth):
        r = (r + np.roll(r, 1, 0) + np.roll(r, 1, 1) + np.roll(r, -1, 0) + np.roll(r, -1, 1)) / 5.0
        r = (r - r.min()) / max(1e-6, r.max() - r.min())
    return lo + (hi - lo) * r


def tex_stone(name="stone", base=(0.50, 0.50, 0.50)):
    n = _noise(name, 0.82, 1.12, 1)
    sp = _rng(name + "s").random((T, T)) < 0.08
    n[sp] *= 0.75
    return _rgba(np.array(base)[None, None, :] * n[..., None])


def tex_dirt(name="dirt"):
    n = _noise(name, 0.78, 1.15)
    return _rgba(np.array([0.53, 0.38, 0.26])[None, None] * n[..., None])


def tex_grass_top(name="grass_top"):
    n = _noise(name, 0.70, 0.95)
    return _rgba(np.stack([n, n, n], -1))


def tex_grass_side(name="grass_side"):
    img = tex_dirt("dirt_gs")
    rng = _rng(name)
    depth = 3 + (rng.random(T) * 2.5).astype(int)
    g = np.array([0.42, 0.62, 0.27])
    for x in range(T):
        for y in range(depth[x]):
            img[y, x, :3] = g * (0.8 + 0.3 * rng.random())
    return img


def tex_sand(name="sand"):
    n = _noise(name, 0.9, 1.05)
    return _rgba(np.array([0.86, 0.81, 0.62])[None, None] * n[..., None])


def tex_gravel(name="gravel"):
    n = _noise(name, 0.6, 1.1)
    return _rgba(np.array([0.55, 0.52, 0.50])[None, None] * n[..., None])


def tex_water(name="water"):
    n = _noise(name, 0.75, 0.95, 2)
    img = _rgba(np.stack([n, n, n], -1), 0.72)
    return img


def tex_lava(name="lava"):
    n = _noise(name, 0.0, 1.0, 2)
    c0 = np.array([0.80, 0.20, 0.02])
    c1 = np.array([1.00, 0.75, 0.15])
    return _rgba(c0[None, None] * (1 - n[..., None]) + c1[None, None] * n[..., None])


def tex_log_side(name="oak_log"):
    rng = _rng(name)
    cols = 0.75 + 0.3 * rng.random(T)
    n = _noise(name + "n", 0.9, 1.05)
    img = _rgba(np.array([0.40, 0.31, 0.19])[None, None] * (cols[None, :] * n)[..., None])
    return img


def tex_log_top(name="oak_log_top"):
    y, x = np.mgrid[0:T, 0:T] + 0.5
    r = np.hypot(x - 8, y - 8)
    rings = 0.85 + 0.15 * np.cos(r * 2.2)
    img = _rgba(np.array([0.70, 0.56, 0.35])[None, None] * rings[..., None])
    edge = (x < 1) | (x > 15) | (y < 1) | (y > 15)
    img[edge, :3] = np.array([0.40, 0.31, 0.19])
    return img


def tex_leaves(name="oak_leaves"):
    n = _noise(name, 0.45, 0.75)
    img = _rgba(np.stack([n, n, n], -1))
    holes = _rng(name + "h").random((T, T)) < 0.28
    img[holes, 3] = 0.0
    return img


def tex_planks(name="oak_planks"):
    n = _noise(name, 0.88, 1.05)
    img = _rgba(np.array([0.64, 0.51, 0.31])[None, None] * n[..., None])
    for y in (3, 7, 11, 15):
        img[y, :, :3] *= 0.65
    return img


def tex_bricks(name="stone_bricks"):
    n = _noise(name, 0.85, 1.05)
    img = _rgba(np.array([0.48, 0.48, 0.48])[None, None] * n[..., None])
    for y in range(T):
        if y % 8 == 7:
            img[y, :, :3] *= 0.6
        off = 0 if (y // 8) % 2 == 0 else 8
        img[y, (off + 7) % T, :3] *= 0.6
        img[y, (off + 15) % T, :3] *= 0.6
    return img


def tex_cobble(name="cobblestone"):
    n = _noise(name, 0.55, 1.15, 1)
    return _rgba(np.array([0.50, 0.50, 0.50])[None, None] * n[..., None])


def tex_smooth_stone(name="smooth_stone"):
    img = _rgba(np.array([0.62, 0.62, 0.62]) * 1.0)
    img[[0, 15], :, :3] *= 0.8
    img[:, [0, 15], :3] *= 0.8
    return img


def tex_glass(name="glass"):
    img = _rgba(np.array([0.85, 0.92, 0.95]), 0.0)
    img[[0, 15], :, 3] = 1.0
    img[:, [0, 15], 3] = 1.0
    for i in range(3, 7):
        img[i, i + 2, 3] = 1.0
        img[i + 1, i + 2, 3] = 1.0
    return img


def tex_stained(name, rgb, a=0.45):
    img = _rgba(np.array(rgb), a)
    img[[0, 15], :, 3] = 0.85
    img[:, [0, 15], 3] = 0.85
    img[[0, 15], :, :3] *= 0.8
    img[:, [0, 15], :3] *= 0.8
    # vanilla-like diagonal streaks (lighter, more opaque), like the clear glass sprite
    for i in range(3, 7):
        for (r, c) in ((i, i + 2), (i + 1, i + 2)):
            img[r, c, :3] = np.minimum(np.array(rgb) * 1.25 + 0.08, 1.0)
            img[r, c, 3] = min(a + 0.3, 1.0)
    return img


def tex_glowstone(name="glowstone"):
    n = _noise(name, 0.0, 1.0, 1)
    c0 = np.array([0.55, 0.40, 0.20])
    c1 = np.array([1.00, 0.90, 0.60])
    return _rgba(c0[None, None] * (1 - n[..., None]) + c1[None, None] * n[..., None])


def tex_torch(name="torch"):
    img = _rgba(np.array([0, 0, 0]), 0.0)
    img[6:16, 7:9, :3] = np.array([0.45, 0.33, 0.18])
    img[6:16, 7:9, 3] = 1.0
    img[6:8, 7:9, :3] = np.array([1.0, 0.85, 0.35])
    img[5:6, 7:9, :3] = np.array([1.0, 0.6, 0.15])
    img[5:6, 7:9, 3] = 1.0
    return img


def tex_lantern(name="lantern"):
    img = _rgba(np.array([0.22, 0.22, 0.25]), 1.0)
    img[3:13, 4:12, :3] = np.array([1.0, 0.72, 0.30])
    img[3:13, 4:12, :3] *= (0.85 + 0.15 * _noise(name, 0, 1)[3:13, 4:12])[..., None]
    img[[2, 13], 3:13, :3] = 0.15
    return img


def tex_soul_lantern(name="soul_lantern"):
    img = _rgba(np.array([0.20, 0.22, 0.26]), 1.0)
    img[3:13, 4:12, :3] = np.array([0.45, 0.88, 0.95])
    img[3:13, 4:12, :3] *= (0.85 + 0.15 * _noise(name, 0, 1)[3:13, 4:12])[..., None]
    img[[2, 13], 3:13, :3] = 0.15
    return img


def tex_flame_torch(name, tip, core):
    img = _rgba(np.array([0, 0, 0]), 0.0)
    img[6:16, 7:9, :3] = np.array([0.45, 0.33, 0.18])
    img[6:16, 7:9, 3] = 1.0
    img[6:8, 7:9, :3] = np.array(core)
    img[5:6, 7:9, :3] = np.array(tip)
    img[5:6, 7:9, 3] = 1.0
    return img


def tex_flower(name, petal):
    img = _rgba(np.array([0, 0, 0]), 0.0)
    img[8:16, 7:9, :3] = np.array([0.25, 0.55, 0.18])
    img[8:16, 7:9, 3] = 1.0
    img[4:8, 6:10, :3] = np.array(petal)
    img[4:8, 6:10, 3] = 1.0
    img[5:7, 7:9, :3] = np.array(petal) * 0.6
    return img


def tex_short_grass(name="short_grass"):
    rng = _rng(name)
    img = _rgba(np.array([0, 0, 0]), 0.0)
    for x in range(1, 15, 2):
        h = 6 + int(rng.random() * 9)
        g = 0.6 + 0.3 * rng.random()
        img[T - h:, x, :3] = g
        img[T - h:, x, 3] = 1.0
    return img


def tex_solid(name, rgb, var=0.08):
    n = _noise(name, 1 - var, 1 + var)
    return _rgba(np.array(rgb)[None, None] * n[..., None])


def tex_rain(name="rain"):
    rng = _rng(name)
    img = _rgba(np.array([0.7, 0.75, 0.85]), 0.0)
    for x in range(0, T, 3):
        y0 = int(rng.random() * T)
        for k in range(6):
            img[(y0 + k) % T, x, 3] = 0.6
    return img


def tex_entity(name="entity"):
    img = tex_solid(name, (0.93, 0.62, 0.62), 0.05)
    img[4:6, 3:5, :3] = 0.1
    img[4:6, 11:13, :3] = 0.1
    return img


TEXTURES = {
    "stone": tex_stone, "dirt": tex_dirt, "grass_top": tex_grass_top, "grass_side": tex_grass_side,
    "sand": tex_sand, "gravel": tex_gravel, "water": tex_water, "lava": tex_lava,
    "oak_log": tex_log_side, "oak_log_top": tex_log_top, "oak_leaves": tex_leaves, "oak_planks": tex_planks,
    "stone_bricks": tex_bricks, "cobblestone": tex_cobble, "smooth_stone": tex_smooth_stone, "glass": tex_glass,
    "red_stained_glass": lambda: tex_stained("red_sg", (0.75, 0.15, 0.12)),
    "blue_stained_glass": lambda: tex_stained("blue_sg", (0.15, 0.25, 0.80)),
    "lime_stained_glass": lambda: tex_stained("lime_sg", (0.35, 0.80, 0.15)),
    "glowstone": tex_glowstone, "torch": tex_torch, "lantern": tex_lantern,
    "poppy": lambda: tex_flower("poppy", (0.85, 0.10, 0.08)),
    "dandelion": lambda: tex_flower("dandelion", (0.98, 0.85, 0.15)),
    "short_grass": tex_short_grass,
    "netherrack": lambda: tex_stone("netherrack", (0.45, 0.17, 0.16)),
    "soul_sand": lambda: tex_solid("soul_sand", (0.33, 0.25, 0.20), 0.15),
    "shroomlight": lambda: tex_solid("shroomlight", (0.98, 0.62, 0.30), 0.1),
    "nether_wart_block": lambda: tex_solid("nether_wart_block", (0.47, 0.03, 0.03), 0.12),
    "end_stone": lambda: tex_stone("end_stone", (0.86, 0.87, 0.62)),
    "obsidian": lambda: tex_solid("obsidian", (0.08, 0.05, 0.13), 0.2),
    "iron_block": lambda: tex_solid("iron_block", (0.85, 0.85, 0.85), 0.03),
    "gold_block": lambda: tex_solid("gold_block", (0.98, 0.82, 0.25), 0.04),
    "rain": tex_rain, "entity": tex_entity,
}

# Sprites of the test structures (cli.py --gallery, --poles, --hall, --furnace, --glass-gallery). The atlas
# places them from tile FIRST_TEST_TILE on, after an empty row, so that no standard sprite gets a new
# neighbour: the atlas has no padding, and mip / bilinear taps near a sprite edge (e.g. the shadow pass's
# 4x4 sprite average of a torch) read the next tile, so the standard scene renders exactly as without them.
FIRST_TEST_TILE = 64
TEST_TEXTURES = {
    # light test scenes
    "soul_lantern": tex_soul_lantern,
    "soul_torch": lambda: tex_flame_torch("soul_torch", (0.30, 0.80, 0.95), (0.55, 0.95, 1.00)),
    "redstone_torch": lambda: tex_flame_torch("redstone_torch", (0.85, 0.05, 0.03), (1.00, 0.20, 0.12)),
    "sea_lantern": lambda: tex_solid("sea_lantern", (0.72, 0.87, 0.84), 0.08),
    "redstone_lamp": lambda: tex_solid("redstone_lamp", (0.95, 0.68, 0.38), 0.10),
    "white_concrete": lambda: tex_solid("white_concrete", (0.81, 0.82, 0.82), 0.02),
    # glass test pieces (--glass-gallery)
    "white_stained_glass": lambda: tex_stained("white_sg", (0.97, 0.97, 0.97), 0.38),
    "orange_stained_glass": lambda: tex_stained("orange_sg", (0.85, 0.48, 0.18)),
    "tinted_glass": lambda: tex_stained("tinted_g", (0.17, 0.13, 0.20), 0.75),
    "ice": lambda: tex_stained("ice", (0.57, 0.71, 0.98), 0.62),
    "slime_block": lambda: tex_stained("slime", (0.45, 0.78, 0.36), 0.72),
    "honey_block": lambda: tex_stained("honey", (0.98, 0.70, 0.18), 0.78),
}
assert len(TEXTURES) <= FIRST_TEST_TILE - 16, "standard sprites must stay clear of the row before FIRST_TEST_TILE"
TEXTURES.update(TEST_TEXTURES)

# rough labPBR material hints per texture: (smoothness, f0 byte, porosity/sss byte, emission mode)
PBR_HINTS = {
    "stone_bricks": (0.25, 10, 40, None), "stone": (0.15, 10, 50, None), "cobblestone": (0.1, 10, 50, None),
    "smooth_stone": (0.45, 10, 20, None), "sand": (0.05, 10, 60, None), "dirt": (0.05, 10, 64, None),
    "glass": (0.95, 10, 0, None), "water": (0.98, 5, 0, None),
    "red_stained_glass": (0.92, 10, 0, None), "blue_stained_glass": (0.92, 10, 0, None),
    "lime_stained_glass": (0.92, 10, 0, None), "white_stained_glass": (0.92, 10, 0, None),
    "orange_stained_glass": (0.92, 10, 0, None), "tinted_glass": (0.92, 10, 0, None),
    "oak_leaves": (0.3, 10, 180, None), "grass_top": (0.15, 10, 140, None), "short_grass": (0.2, 10, 170, None),
    "poppy": (0.2, 10, 160, None), "dandelion": (0.2, 10, 160, None),
    "glowstone": (0.2, 10, 0, "bright"), "lava": (0.4, 10, 0, "all"), "torch": (0.2, 10, 0, "bright"),
    "lantern": (0.6, 230, 0, "bright"), "shroomlight": (0.2, 10, 0, "all"),
    "soul_lantern": (0.6, 230, 0, "bright"), "soul_torch": (0.2, 10, 0, "bright"), "redstone_torch": (0.2, 10, 0, "bright"),
    "sea_lantern": (0.3, 10, 0, "all"), "redstone_lamp": (0.3, 10, 0, "all"),
    "iron_block": (0.75, 230, 0, None), "gold_block": (0.8, 231, 0, None), "obsidian": (0.7, 10, 0, None),
}


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def fill_transparent(img):
    """Give fully transparent texels (a == 0) the alpha-weighted mean colour of the sprite, averaged
    in linear light, keeping a == 0. Emulates Sodium's sprite pre-processing (vanilla does the same
    for cutout sprites): without it the RGBA8 box-filtered mipmaps average the black RGB of the
    transparent texels into the visible ones, so distant cutout plants (grass tufts, flowers,
    leaves) turn up to ~2x too dark in the mip levels. Returns a new array."""
    a = img[..., 3]
    clear = a <= 0.0
    w = float(a.sum())
    if not clear.any() or w <= 0.0:
        return img
    out = img.copy()
    mean_lin = (_srgb_to_linear(img[..., :3]) * a[..., None]).sum((0, 1)) / w
    out[clear, :3] = _linear_to_srgb(mean_lin).astype(np.float32)
    return out


class Atlas:
    def __init__(self, names, pbr=False):
        self.grid = 16
        self.size = self.grid * T
        self.index = {}
        albedo = np.zeros((self.size, self.size, 4), np.float32)
        normal = np.zeros_like(albedo)
        normal[...] = (127 / 255.0, 127 / 255.0, 1.0, 1.0)   # Iris PBRType.NORMAL default 0x7F7FFFFF
        spec = np.zeros_like(albedo)
        k_std, k_test = 0, FIRST_TEST_TILE
        for n in names:
            if n in TEST_TEXTURES:
                i, k_test = k_test, k_test + 1
            else:
                i, k_std = k_std, k_std + 1
            tx, ty = i % self.grid, i // self.grid
            img = TEXTURES[n]()
            sl = (slice(ty * T, ty * T + T), slice(tx * T, tx * T + T))
            # colour-filled before upload (mipmaps); the PBR maps below are derived from the
            # unfilled sprite so their edge slopes stay as before
            albedo[sl] = fill_transparent(img)
            self.index[n] = (tx, ty)
            if pbr:
                lum = img[..., :3] @ np.array([0.3, 0.6, 0.1], np.float32)
                dx = np.roll(lum, -1, 1) - np.roll(lum, 1, 1)
                dy = np.roll(lum, -1, 0) - np.roll(lum, 1, 0)
                nv = np.stack([-dx * 2.0, dy * 2.0, np.ones_like(lum)], -1)
                nv /= np.linalg.norm(nv, axis=-1, keepdims=True)
                normal[sl][..., 0:2] = nv[..., 0:2] * 0.5 + 0.5
                normal[sl][..., 2] = 1.0           # AO
                normal[sl][..., 3] = 0.85 + 0.15 * lum  # height
                h = PBR_HINTS.get(n)
                if h:
                    sm, f0, ps, em = h
                    spec[sl][..., 0] = sm
                    spec[sl][..., 1] = f0 / 255.0
                    spec[sl][..., 2] = ps / 255.0
                    if em == "all":
                        spec[sl][..., 3] = 254 / 255.0
                    elif em == "bright":
                        spec[sl][..., 3] = np.where(lum > 0.6, 254 / 255.0, 0.0)
        self.albedo = albedo
        self.normal = normal
        self.specular = spec

    def uv_rect(self, name, u0=0.0, v0=0.0, u1=1.0, v1=1.0):
        tx, ty = self.index[name]
        s = 1.0 / self.grid
        return (tx + u0) * s, (ty + v0) * s, (tx + u1) * s, (ty + v1) * s


def to_u8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def vanilla_lightmap(sky_brightness=1.0, dim="overworld", night_vision=0.0):
    """16x16 RGB lightmap similar to vanilla LightTexture (x = block, y = sky)."""
    lm = np.zeros((16, 16, 4), np.float32)
    lvl = np.arange(16) / 15.0
    curve = lvl / (4.0 - 3.0 * lvl)  # vanilla brightness curve (ambient 0)
    for s in range(16):
        for b in range(16):
            sky = curve[s] * (sky_brightness * 0.95 + 0.05)
            blk = curve[b] * 1.5
            col = np.array([sky * 0.95 + blk, sky * 0.95 + blk * (0.7 + 0.3 * curve[b] ** 2),
                            sky + blk * (0.6 + 0.4 * curve[b] ** 4)])
            if dim == "nether":
                col = col * 0.9 + 0.1
            col = np.clip(col, 0, 1)
            lm[s, b, :3] = col
            lm[s, b, 3] = 1.0
    return lm
