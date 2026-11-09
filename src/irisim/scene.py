"""Synthetic Minecraft-like test world, meshed on the CPU like Sodium would.

World: 96 x 48 x 96 blocks (x, y, z), placed at WORLD_ORIGIN in world space.
Vertex layout (float32, interleaved, 26 floats):
  pos3 (relative to the region origin = WORLD_ORIGIN), color4 (biome tint rgb,
  vanilla AO in alpha when separateAo), uv0 2, uv2 2 (lightmap, vanilla
  encoding: level*16), normal3, mc_Entity2 (block id, render layer),
  mc_midTexCoord2, at_tangent4, at_midBlock4 (xyz = (centre - vertex) * 64,
  w = emission).
Every quad is 4 consecutive vertices (gl_VertexID % 4 == 0 starts a quad),
drawn with an index buffer as two triangles.
"""
import numpy as np

from . import textures

SX, SY, SZ = 96, 48, 96
WORLD_ORIGIN = np.array([936, 32, -1096], np.int64)   # world coords of block index (0,0,0); cli.py --origin-dy shifts y
SEA = 30                                               # index y of the water surface block (world y 62)
VERTEX_FLOATS = 26
LAYERS = ["solid", "cutout_mipped", "cutout", "translucent"]

# ---------------------------------------------------------------- block registry
#            name                 shape        layer           textures (top, side, bottom)                      tint      emit opaque  lightOpacity
BLOCKS = [
    ("air",                       "none",      None,           None,                                              None,     0,   False, 0),
    ("stone",                     "cube",      "solid",        ("stone",) * 3,                                    None,     0,   True,  15),
    ("dirt",                      "cube",      "solid",        ("dirt",) * 3,                                     None,     0,   True,  15),
    ("grass_block",               "cube",      "solid",        ("grass_top", "grass_side", "dirt"),               "grass_top", 0, True,  15),
    ("sand",                      "cube",      "solid",        ("sand",) * 3,                                     None,     0,   True,  15),
    ("water",                     "liquid",    "translucent",  ("water",) * 3,                                    "water",  0,   False, 1),
    ("oak_log",                   "cube",      "solid",        ("oak_log_top", "oak_log", "oak_log_top"),         None,     0,   True,  15),
    ("oak_leaves",                "cube",      "cutout_mipped", ("oak_leaves",) * 3,                              "foliage", 0,  False, 1),
    ("stone_bricks",              "cube",      "solid",        ("stone_bricks",) * 3,                             None,     0,   True,  15),
    ("glass",                     "cube",      "cutout",       ("glass",) * 3,                                    None,     0,   False, 0),
    ("torch",                     "torch",     "cutout",       ("torch",) * 3,                                    None,     14,  False, 0),
    ("lantern",                   "lantern",   "cutout",       ("lantern",) * 3,                                  None,     15,  False, 0),
    ("glowstone",                 "cube",      "solid",        ("glowstone",) * 3,                                None,     15,  True,  15),
    ("lava",                      "liquid",    "solid",        ("lava",) * 3,                                     None,     15,  False, 15),
    ("red_stained_glass",         "cube",      "translucent",  ("red_stained_glass",) * 3,                        None,     0,   False, 0),
    ("blue_stained_glass",        "cube",      "translucent",  ("blue_stained_glass",) * 3,                       None,     0,   False, 0),
    ("lime_stained_glass",        "cube",      "translucent",  ("lime_stained_glass",) * 3,                       None,     0,   False, 0),
    ("smooth_stone_slab",         "slab",      "solid",        ("smooth_stone",) * 3,                             None,     0,   False, 15),
    ("poppy",                     "cross",     "cutout",       ("poppy",) * 3,                                    None,     0,   False, 0),
    ("dandelion",                 "cross",     "cutout",       ("dandelion",) * 3,                                None,     0,   False, 0),
    ("short_grass",               "cross",     "cutout",       ("short_grass",) * 3,                              "grass",  0,   False, 0),
    ("oak_planks",                "cube",      "solid",        ("oak_planks",) * 3,                               None,     0,   True,  15),
    ("cobblestone",               "cube",      "solid",        ("cobblestone",) * 3,                              None,     0,   True,  15),
    ("gravel",                    "cube",      "solid",        ("gravel",) * 3,                                   None,     0,   True,  15),
    ("netherrack",                "cube",      "solid",        ("netherrack",) * 3,                               None,     0,   True,  15),
    ("soul_sand",                 "cube",      "solid",        ("soul_sand",) * 3,                                None,     0,   True,  15),
    ("shroomlight",               "cube",      "solid",        ("shroomlight",) * 3,                              None,     15,  True,  15),
    ("nether_wart_block",         "cube",      "solid",        ("nether_wart_block",) * 3,                        None,     0,   True,  15),
    ("end_stone",                 "cube",      "solid",        ("end_stone",) * 3,                                None,     0,   True,  15),
    ("obsidian",                  "cube",      "solid",        ("obsidian",) * 3,                                 None,     0,   True,  15),
    ("iron_block",                "cube",      "solid",        ("iron_block",) * 3,                               None,     0,   True,  15),
    ("gold_block",                "cube",      "solid",        ("gold_block",) * 3,                               None,     0,   True,  15),
    # coloured emitters and a neutral wall for the light test scenes (--gallery, --poles, --furnace, --hall)
    ("soul_lantern",              "lantern",   "cutout",       ("soul_lantern",) * 3,                             None,     10,  False, 0),
    ("soul_torch",                "torch",     "cutout",       ("soul_torch",) * 3,                               None,     10,  False, 0),
    ("redstone_torch",            "torch",     "cutout",       ("redstone_torch",) * 3,                           None,     7,   False, 0),
    ("sea_lantern",               "cube",      "solid",        ("sea_lantern",) * 3,                              None,     15,  True,  15),
    ("redstone_lamp",             "cube",      "solid",        ("redstone_lamp",) * 3,                            None,     15,  True,  15),
    ("white_concrete",            "cube",      "solid",        ("white_concrete",) * 3,                           None,     0,   True,  15),
    # glass test pieces (--glass-gallery). Vanilla render layers: glass_pane cutout_mipped, stained / tinted glass,
    # ice, slime and honey translucent; block.properties layer.* lines override them like Iris does (cli.py)
    ("glass_pane",                "pane",      "cutout_mipped", ("glass",) * 3,                                   None,     0,   False, 0),
    ("red_stained_glass_pane",    "pane",      "translucent",  ("red_stained_glass",) * 3,                        None,     0,   False, 0),
    ("blue_stained_glass_pane",   "pane",      "translucent",  ("blue_stained_glass",) * 3,                       None,     0,   False, 0),
    ("white_stained_glass",       "cube",      "translucent",  ("white_stained_glass",) * 3,                      None,     0,   False, 0),
    ("tinted_glass",              "cube",      "translucent",  ("tinted_glass",) * 3,                             None,     0,   False, 15),
    ("orange_stained_glass",      "cube",      "translucent",  ("orange_stained_glass",) * 3,                     None,     0,   False, 0),
    ("ice",                       "cube",      "translucent",  ("ice",) * 3,                                      None,     0,   False, 1),
    ("slime_block",               "cube",      "translucent",  ("slime_block",) * 3,                              None,     0,   False, 1),
    ("honey_block",               "cube",      "translucent",  ("honey_block",) * 3,                              None,     0,   False, 1),
]
BID = {b[0]: i for i, b in enumerate(BLOCKS)}
BLOCK_STATE = {"smooth_stone_slab": {"type": "bottom"}, "lantern": {"hanging": "false"},
               "soul_lantern": {"hanging": "false"}, "redstone_torch": {"lit": "true"}, "redstone_lamp": {"lit": "true"},
               "soul_torch": {}, "oak_log": {"axis": "y"},
               "water": {"level": "0"}, "lava": {"level": "0"}, "torch": {}, "short_grass": {}}
TINTS = {"grass": (0.569, 0.741, 0.349), "grass_top": (0.569, 0.741, 0.349), "foliage": (0.467, 0.671, 0.184),
         "water": (0.247, 0.463, 0.894)}

# Built-in id table (only used with --builtin-ids when block.properties is missing):
BUILTIN_IDS = {"stone": 114, "dirt": 113, "grass_block": 112, "sand": 106, "water": 20, "oak_log": 109,
               "oak_leaves": 13, "stone_bricks": 104, "glass": 21, "torch": 40, "lantern": 43, "glowstone": 45,
               "lava": 50, "red_stained_glass": 21, "blue_stained_glass": 21, "lime_stained_glass": 21,
               "smooth_stone_slab": 80, "poppy": 10, "dandelion": 10, "short_grass": 10, "oak_planks": 109,
               "cobblestone": 114, "gravel": 106, "netherrack": 118, "soul_sand": 118, "shroomlight": 47,
               "nether_wart_block": 126, "end_stone": 119, "obsidian": 111, "iron_block": 100, "gold_block": 101}

NB = len(BLOCKS)
SHAPE = [b[1] for b in BLOCKS]
LAYER = [b[2] for b in BLOCKS]
EMIT = np.array([b[5] for b in BLOCKS], np.int32)
OPAQUE = np.array([b[6] for b in BLOCKS], bool)
LIGHT_OPACITY = np.array([b[7] for b in BLOCKS], np.int32)
IS_CUBEISH = np.array([b[1] in ("cube", "liquid") for b in BLOCKS], bool)
# faces of these blocks are hidden against the same block (water/water, glass/glass)
CULL_SAME = np.array([b[1] == "liquid" or b[0].endswith("glass") for b in BLOCKS], bool)

DIRS = [  # normal, tangent (u direction), "down" (v direction)
    ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
]


def _hash2(x, z, seed):
    h = (x * 374761393 + z * 668265263 + seed * 1442695040888963407) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


class World:
    def __init__(self, dimension="overworld", seed=1, test_step=False, extras=None):
        self.dim = dimension
        self.test_step = test_step   # --test-step: 1-high stone step in the cave chamber (hand-light test)
        # optional test structures (Overworld only), see cli.py: {"glass_gallery": bool, "village": bool,
        # "gallery": [block, ...], "poles": [block, ...], "hall": (wall, light), "furnace": (wall, light)};
        # a missing / empty entry leaves the standard scene untouched
        self.extras = dict(extras or {})
        self.blocks = np.zeros((SX, SY, SZ), np.uint8)
        self.rng = np.random.default_rng(seed)
        self.features = {}
        self.pane_axis = {}          # pane blocks: (x, y, z) -> "x" (spans x, thin in z) or "z"
        self._generate()
        self.compute_light()

    # ------------------------------------------------------------ generation
    def _generate(self):
        B = self.blocks
        xs, zs = np.meshgrid(np.arange(SX), np.arange(SZ), indexing="ij")
        h = (33.0 + 1.6 * np.sin(xs * 0.13) * np.cos(zs * 0.11) + 1.2 * np.sin((xs + 2 * zs) * 0.05)
             + 0.8 * np.cos(xs * 0.31 + 1.0) * np.sin(zs * 0.27))
        # lake bowl
        lake_c = (28, 58)
        d = np.hypot(xs - lake_c[0], zs - lake_c[1])
        bowl = SEA - 4 + (d / 13.0) ** 2 * 6
        h = np.where(d < 16, np.minimum(h, bowl), h)
        # hill with a cave
        hill_c = (70, 80)
        dh = np.hypot(xs - hill_c[0], (zs - hill_c[1]) * 0.8)
        h = h + 13.0 * np.clip(1.0 - (dh / 17.0) ** 2, 0, 1)
        # flatten the house plot
        hx0, hz0 = 54, 40
        h[hx0 - 2:hx0 + 11, hz0 - 2:hz0 + 11] = 34
        h = np.clip(np.floor(h).astype(int), 4, SY - 6)
        self.height = h
        is_shore = np.zeros_like(h, bool)
        top = "grass_block"
        sub = "dirt"
        base = "stone"
        if self.dim == "nether":
            top, sub, base = "netherrack", "netherrack", "netherrack"
        elif self.dim == "end":
            top, sub, base = "end_stone", "end_stone", "end_stone"
        for x in range(SX):
            for z in range(SZ):
                hh = h[x, z]
                B[x, :hh - 3, z] = BID[base]
                B[x, hh - 3:hh - 1, z] = BID[sub]
                shore = hh - 1 <= SEA + 1 and d[x, z] < 18.5
                is_shore[x, z] = shore
                if self.dim == "overworld":
                    B[x, hh - 1, z] = BID["sand" if shore else top]
                elif self.dim == "nether":
                    B[x, hh - 1, z] = BID["soul_sand" if shore else top]
                else:
                    B[x, hh - 1, z] = BID[top]
                if hh <= SEA and d[x, z] < 18.5:
                    fluid = "lava" if self.dim == "nether" else "water"
                    if self.dim != "end":
                        B[x, hh:SEA + 1, z] = BID[fluid]
        self.surface = h  # first air block above ground (before structures)

        if self.dim == "overworld":
            self._house(hx0, hz0)
            self._stained_glass_wall(42, 38)
            if self.extras.get("glass_gallery"):
                self._glass_gallery()
            for (tx, tz, th) in [(14, 24, 5), (42, 26, 6), (84, 28, 5), (16, 84, 6), (88, 52, 5), (8, 44, 4)]:
                self._tree(tx, tz, th)
            self._plants()
        self._cave(hill_c)
        self._lava_pool(52, 82)
        self._glowstone_pillar(38, 46)
        if self.dim == "nether":
            for (x, z) in [(20, 20), (80, 30), (60, 60)]:
                g = self.surface[x, z]
                B[x, g:g + 3, z] = BID["nether_wart_block"]
                B[x, g + 3, z] = BID["shroomlight"]
        if self.dim == "end":
            for (x, z) in [(20, 30), (75, 20)]:
                g = self.surface[x, z]
                B[x - 1:x + 2, g:g + 14, z - 1:z + 2] = BID["obsidian"]
        # torches by the lake
        for (x, z) in [(40, 50), (34, 44), (18, 70)]:
            g = self._ground(x, z)
            if B[x, g - 1, z] not in (BID["water"], BID["lava"]):
                B[x, g, z] = BID["torch"]
        if self.dim == "overworld":
            self._light_extras()
        if self.dim == "nether":
            B[:, SY - 3:, :] = BID["netherrack"]   # ceiling

    def _ground(self, x, z):
        col = self.blocks[x, :, z]
        solid = np.nonzero((col != 0) & np.isin(col, [BID[n] for n in BID if SHAPE[BID[n]] == "cube"]))[0]
        return int(solid.max()) + 1 if len(solid) else 1

    def _house(self, x0, z0):
        B = self.blocks
        g = 34  # floor top
        W, D, H = 9, 9, 4
        B[x0:x0 + W, g - 1, z0:z0 + D] = BID["oak_planks"]
        B[x0:x0 + W, g:g + H + 2, z0:z0 + D] = 0
        for y in range(g, g + H):
            B[x0, y, z0:z0 + D] = BID["stone_bricks"]
            B[x0 + W - 1, y, z0:z0 + D] = BID["stone_bricks"]
            B[x0:x0 + W, y, z0] = BID["stone_bricks"]
            B[x0:x0 + W, y, z0 + D - 1] = BID["stone_bricks"]
        # roof: stone bricks rim + slab layer
        B[x0:x0 + W, g + H, z0:z0 + D] = BID["stone_bricks"]
        B[x0 + 1:x0 + W - 1, g + H, z0 + 1:z0 + D - 1] = BID["smooth_stone_slab"]
        # windows
        for k in (2, 6):
            B[x0 + k, g + 1:g + 3, z0] = BID["glass"]
            B[x0 + k, g + 1:g + 3, z0 + D - 1] = BID["glass"]
            B[x0, g + 1:g + 3, z0 + k] = BID["glass"]
            B[x0 + W - 1, g + 1:g + 3, z0 + k] = BID["glass"]
        # door (north side, facing -z)
        B[x0 + 4, g:g + 2, z0] = 0
        # lanterns: one inside, one by the door
        B[x0 + 4, g, z0 + 5] = BID["lantern"]
        B[x0 + 6, g, z0 - 1] = BID["lantern"]
        B[x0 + 1, g + 2, z0 + 1] = BID["torch"]
        # slab path towards the lake
        for z in range(z0 - 8, z0):
            B[x0 + 4, g, z] = BID["smooth_stone_slab"]
            B[x0 + 4, g - 1, z] = BID["gravel"]
        self.features["house"] = (x0, g, z0, W, H, D)

    def _light_extras(self):
        """Light test structures: --village, --gallery, --poles, --hall, --furnace (see cli.py and
        README). --gallery and --poles share the platform x 0..41, z 0..29; --hall and --furnace the area
        x 60..91, z 2..33 (cli.py refuses both of a pair)."""
        B = self.blocks
        ex = self.extras
        if ex.get("village"):
            # lamp posts around the house: 2-block oak posts with a lantern or torch on top
            for (x, z, kind) in [(55, 34, "lantern"), (61, 34, "lantern"), (55, 28, "torch"), (61, 28, "lantern"),
                                 (49, 31, "lantern"), (67, 31, "torch"), (52, 37, "torch"), (64, 37, "lantern")]:
                g = self._ground(x, z)
                B[x, g:g + 2, z] = BID["oak_log"]
                B[x, g + 2, z] = BID[kind]
            self.features["village"] = True
        if ex.get("gallery"):
            # coloured-light gallery: up to 9 booths (white concrete, 3 x 3 x 3 inside, open towards -z) on a
            # flat white platform, one emitter on the floor of each (booth i spans x 2+4i .. 6+4i, z 20..24)
            g = 34
            B[0:42, :g, 0:30] = np.where(B[0:42, :g, 0:30] == 0, BID["stone"], B[0:42, :g, 0:30])
            B[0:42, g:, 0:30] = 0
            B[0:42, g - 1, 0:30] = BID["white_concrete"]
            for i, kind in enumerate(ex["gallery"]):
                x0 = 2 + 4 * i
                B[x0:x0 + 5, g:g + 3, 24] = BID["white_concrete"]          # back wall
                B[x0, g:g + 3, 20:25] = BID["white_concrete"]              # side walls
                B[x0 + 4, g:g + 3, 20:25] = BID["white_concrete"]
                B[x0:x0 + 5, g + 3, 20:25] = BID["white_concrete"]         # ceiling
                B[x0 + 2, g, 22] = BID[kind]
            self.features["gallery"] = (2, g, 20)
        if ex.get("poles"):
            # up to 5 lights on 5-block oak poles in the open (seen against the night sky), 8 blocks apart
            g = 34
            B[0:42, :g, 0:30] = np.where(B[0:42, :g, 0:30] == 0, BID["stone"], B[0:42, :g, 0:30])
            B[0:42, g:, 0:30] = 0
            B[0:42, g - 1, 0:30] = BID["grass_block"]
            for i, kind in enumerate(ex["poles"]):
                x = 6 + 8 * i
                B[x, g:g + 5, 20] = BID["oak_log"]
                B[x, g + 5, 20] = BID[kind]
            self.features["poles"] = (6, g + 5, 20)
        if ex.get("hall"):
            # closed hall (interior 30 x 7 x 30, floating above the terrain) lit by 4 emitters on the floor
            wall, light = ex["hall"]
            x0, y0, z0 = 60, 38, 2
            B[x0:x0 + 32, y0:y0 + 9, z0:z0 + 32] = BID[wall]
            B[x0 + 1:x0 + 31, y0 + 1:y0 + 8, z0 + 1:z0 + 31] = 0
            for (dx, dz) in [(9, 9), (21, 9), (9, 21), (21, 21)]:
                B[x0 + 1 + dx, y0 + 1, z0 + 1 + dz] = BID[light]
            self.features["hall"] = (x0, y0, z0)
        if ex.get("furnace"):
            # closed room (interior 7 x 5 x 7, floating) with one emitter in the middle of the floor: a
            # "furnace" test for GI energy conservation (indirect light only from the room's walls)
            wall, light = ex["furnace"]
            x0, y0, z0 = 80, 38, 4
            B[x0:x0 + 9, y0:y0 + 7, z0:z0 + 9] = BID[wall]
            B[x0 + 1:x0 + 8, y0 + 1:y0 + 6, z0 + 1:z0 + 8] = 0
            B[x0 + 4, y0 + 1, z0 + 4] = BID[light]
            self.features["furnace"] = (x0, y0, z0)

    def _glass_gallery(self):
        """Glass test pieces (--glass-gallery) on a cobblestone floor (x 36..53, z 35..42) around the
        stained-glass wall (x 42..44, z 38), seen by --camera glass / glass2 / glass3 / glass4:
        3-high panes (clear, red, blue) at x 38..40, a thick clear glass block cluster (2 wide x 3 high
        x 2 deep) at x 46..47, z 37..38, white, orange and tinted glass blocks (2 high) at x 49..51,
        2 ice blocks at x 37, honey (below) and slime (above) at x 36, and props behind them (iron
        block, gold block, lantern on a post at z 41). In the house (--camera interior / interior2)
        the west wall gets a clear pane window (z0+2) and a red (lower) / blue (upper) stained glass
        block window (z0+4); the south wall a red / blue stained pane window (x0+2)."""
        B = self.blocks
        g = int(self.surface[42, 38])          # terrain top + 1 (the stained-glass wall stands on it)
        wall = B[42:45, g:g + 3, 38].copy()
        for x in range(36, 54):
            for z in range(35, 43):
                B[x, max(g - 4, 0):g - 1, z] = np.where(B[x, max(g - 4, 0):g - 1, z] == 0, BID["dirt"], B[x, max(g - 4, 0):g - 1, z])
                B[x, g - 1, z] = BID["cobblestone"]
                B[x, g:g + 6, z] = 0
        B[42:45, g:g + 3, 38] = wall
        for i, n in enumerate(["glass_pane", "red_stained_glass_pane", "blue_stained_glass_pane"]):
            for y in range(3):
                B[38 + i, g + y, 38] = BID[n]
                self.pane_axis[(38 + i, g + y, 38)] = "x"
        B[46:48, g:g + 3, 37:39] = BID["glass"]
        for i, n in enumerate(["white_stained_glass", "orange_stained_glass", "tinted_glass"]):
            for y in range(2):
                B[49 + i, g + y, 38] = BID[n]
        B[37, g:g + 2, 38] = BID["ice"]
        B[36, g, 38] = BID["honey_block"]
        B[36, g + 1, 38] = BID["slime_block"]
        # props behind the glass: a lantern on a post and a gold block (edges to see refraction against)
        B[47, g, 41] = BID["oak_log"]
        B[47, g + 1, 41] = BID["lantern"]
        B[44, g, 41] = BID["gold_block"]
        B[39, g, 41] = BID["iron_block"]
        if "house" in self.features:
            x0, hg, z0, W, H, D = self.features["house"]
            # west wall: z0+2 clear panes, z0+4 red (lower) / blue (upper) stained glass blocks,
            # z0+6 clear glass blocks (as before); south wall z0+D-1, x0+2: red / blue stained panes
            for y in range(hg + 1, hg + 3):
                B[x0, y, z0 + 2] = BID["glass_pane"]
                self.pane_axis[(x0, y, z0 + 2)] = "z"
                B[x0, y, z0 + 4] = BID["red_stained_glass"] if y == hg + 1 else BID["blue_stained_glass"]
                B[x0 + 2, y, z0 + D - 1] = BID["red_stained_glass_pane"] if y == hg + 1 else BID["blue_stained_glass_pane"]
                self.pane_axis[(x0 + 2, y, z0 + D - 1)] = "x"

    def _stained_glass_wall(self, x0, z0):
        B = self.blocks
        g = self._ground(x0, z0)
        for i, n in enumerate(["red_stained_glass", "blue_stained_glass", "lime_stained_glass"]):
            for y in range(3):
                B[x0 + i, g + y, z0] = BID[n]
        B[x0 - 1:x0 + 4, g - 1, z0] = BID["cobblestone"]

    def _tree(self, x, z, height):
        B = self.blocks
        g = self._ground(x, z)
        if g + height + 3 >= SY:
            return
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                for dy in range(height - 3, height + 2):
                    r = abs(dx) + abs(dz) + max(0, dy - height + 1) * 2
                    if r <= 3 and 0 <= x + dx < SX and 0 <= z + dz < SZ and B[x + dx, g + dy, z + dz] == 0:
                        if _hash2(x + dx, z + dz + dy * 31, 7) > 0.08:
                            B[x + dx, g + dy, z + dz] = BID["oak_leaves"]
        B[x, g:g + height, z] = BID["oak_log"]
        B[x, g - 1, z] = BID["dirt"]

    def _plants(self):
        B = self.blocks
        for x in range(1, SX - 1):
            for z in range(1, SZ - 1):
                g = self._ground(x, z)
                if g >= SY - 1 or B[x, g - 1, z] != BID["grass_block"] or B[x, g, z] != 0:
                    continue
                r = _hash2(x, z, 3)
                if r < 0.10:
                    B[x, g, z] = BID["short_grass"]
                elif r < 0.12:
                    B[x, g, z] = BID["poppy"]
                elif r < 0.135:
                    B[x, g, z] = BID["dandelion"]

    def _cave(self, c):
        B = self.blocks
        cx, cz = c
        g = int(self.surface[cx, cz - 21])
        y0 = max(g, SEA + 2)
        # tunnel from the north face of the hill going south (+z), then a chamber
        for z in range(cz - 16, cz + 4):
            for x in range(cx - 1, cx + 2):
                B[x, y0:y0 + 3, z] = 0
            B[cx - 1:cx + 2, y0 - 1, z] = BID["stone"] if self.dim == "overworld" else int(B[cx, y0 - 2, z])
        B[cx - 4:cx + 5, y0:y0 + 5, cz + 2:cz + 9] = 0
        B[cx - 4:cx + 5, y0 - 1, cz + 2:cz + 9] = BID["gravel" if self.dim == "overworld" else "netherrack"]
        B[cx - 3, y0, cz + 7] = BID["torch"]
        B[cx + 3, y0, cz + 4] = BID["iron_block"]
        B[cx + 3, y0, cz + 5] = BID["gold_block"]
        if self.test_step:
            # hand-light test (r6wind-d1): a 1-block-high stone step (top at y0 + 1) in the chamber, in front of
            # a camera at --pos 69.3,34.27,85.5 --yaw -90 (crouched eye 0.27 above the step top)
            B[cx + 1:cx + 3, y0, cz + 4:cz + 7] = BID["stone"]
        self.features["cave"] = (cx, y0, cz)

    def _lava_pool(self, x0, z0):
        B = self.blocks
        g = self._ground(x0, z0)
        B[x0 - 2:x0 + 3, g - 2, z0 - 2:z0 + 3] = BID["stone" if self.dim != "nether" else "netherrack"]
        B[x0 - 1:x0 + 2, g - 1, z0 - 1:z0 + 2] = BID["lava"]
        B[x0 - 1:x0 + 2, g:g + 2, z0 - 1:z0 + 2] = 0
        self.features["lava"] = (x0, g, z0)

    def _glowstone_pillar(self, x, z):
        B = self.blocks
        g = self._ground(x, z)
        B[x, g:g + 2, z] = BID["cobblestone"]
        B[x, g + 2, z] = BID["glowstone"]
        self.features["glowstone"] = (x, g + 2, z)

    def clear_plants_near(self, p, r):
        x0, y0, z0 = (int(np.floor(v)) for v in p)
        sl = (slice(max(0, x0 - r), x0 + r + 1), slice(max(0, y0 - 3), y0 + 2), slice(max(0, z0 - r), z0 + r + 1))
        sub = self.blocks[sl]
        sub[np.isin(sub, [BID["poppy"], BID["dandelion"], BID["short_grass"]])] = 0

    # ------------------------------------------------------------ light
    def compute_light(self):
        B = self.blocks
        opac = LIGHT_OPACITY[B]
        opaque_l = opac >= 15
        emit = EMIT[B]
        sky = np.zeros(B.shape, np.int32)
        if self.dim == "overworld":
            # direct skylight straight down
            level = np.full((SX, SZ), 15, np.int32)
            for y in range(SY - 1, -1, -1):
                o = opac[:, y, :]
                level = np.where(o >= 15, 0, np.maximum(level - o, 0))
                sky[:, y, :] = level
            sky = self._flood(sky, opac, opaque_l)
        blk = self._flood(emit.copy(), opac, opaque_l, keep=emit)
        self.sky = sky.astype(np.uint8)
        self.blk = blk.astype(np.uint8)

    @staticmethod
    def _flood(L, opac, opaque_l, keep=None):
        for _ in range(15):
            P = np.pad(L, 1, mode="constant", constant_values=0)
            nb = np.maximum.reduce([P[2:, 1:-1, 1:-1], P[:-2, 1:-1, 1:-1], P[1:-1, 2:, 1:-1],
                                    P[1:-1, :-2, 1:-1], P[1:-1, 1:-1, 2:], P[1:-1, 1:-1, :-2]])
            cand = nb - np.maximum(1, opac)   # vanilla: decrement max(1, opacity) per step
            newL = np.where(opaque_l, 0, np.maximum(L, cand))
            if keep is not None:
                newL = np.maximum(newL, keep)
            if np.array_equal(newL, L):
                break
            L = newL
        return np.clip(L, 0, 15)

    def light_at(self, p):
        x, y, z = (int(np.floor(v)) for v in p)
        if 0 <= x < SX and 0 <= y < SY and 0 <= z < SZ:
            return int(self.blk[x, y, z]), int(self.sky[x, y, z])
        return 0, 15 if self.dim == "overworld" else 0

    def block_at(self, p):
        x, y, z = (int(np.floor(v)) for v in p)
        if 0 <= x < SX and 0 <= y < SY and 0 <= z < SZ:
            return BLOCKS[self.blocks[x, y, z]][0]
        return "air"


# ==================================================================== meshing
class MeshBuilder:
    def __init__(self, atlas, ids, separate_ao=True):
        self.atlas = atlas
        self.ids = ids
        self.separate_ao = separate_ao
        self.parts = {l: [] for l in LAYERS}

    def add(self, layer, verts):
        self.parts[layer].append(verts)

    def finish(self):
        out = {}
        for l, parts in self.parts.items():
            if parts:
                v = np.concatenate(parts).astype(np.float32)
            else:
                v = np.zeros((0, VERTEX_FLOATS), np.float32)
            out[l] = v
        return out


def _tangents(pos, uv, normal):
    """Iris NormalHelper.computeTangent on the first triangle of each quad. pos (Q,4,3) uv (Q,4,2) n (Q,3)."""
    e1 = pos[:, 1] - pos[:, 0]
    e2 = pos[:, 2] - pos[:, 0]
    du1 = uv[:, 1, 0] - uv[:, 0, 0]
    dv1 = uv[:, 1, 1] - uv[:, 0, 1]
    du2 = uv[:, 2, 0] - uv[:, 0, 0]
    dv2 = uv[:, 2, 1] - uv[:, 0, 1]
    den = du1 * dv2 - du2 * dv1
    f = np.where(den == 0, 1.0, 1.0 / np.where(den == 0, 1.0, den))
    t = f[:, None] * (dv2[:, None] * e1 - dv1[:, None] * e2)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    b = f[:, None] * (-du2[:, None] * e1 + du1[:, None] * e2)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    pb = np.cross(t, normal)
    w = np.where(np.sum(b * pb, axis=1) < 0, -1.0, 1.0)
    return np.concatenate([t, w[:, None]], axis=1)


def _assemble(pos, color, uv, lm, normal, ent, center, emission):
    """Build (Q*4, 26) vertex array. pos/uv/lm (Q,4,k), color (Q,4,4), normal (Q,3), ent (Q,2), center (Q,3)."""
    Q = pos.shape[0]
    if Q == 0:
        return np.zeros((0, VERTEX_FLOATS), np.float32)
    v = np.zeros((Q, 4, VERTEX_FLOATS), np.float32)
    v[:, :, 0:3] = pos
    v[:, :, 3:7] = color
    v[:, :, 7:9] = uv
    v[:, :, 9:11] = lm
    v[:, :, 11:14] = normal[:, None, :]
    v[:, :, 14:16] = ent[:, None, :]
    v[:, :, 16:18] = uv.mean(axis=1)[:, None, :]
    v[:, :, 18:22] = _tangents(pos, uv, normal)[:, None, :]
    v[:, :, 22:25] = (center[:, None, :] - pos) * 64.0
    v[:, :, 25] = emission[:, None]
    return v.reshape(Q * 4, VERTEX_FLOATS)


def _fix_winding(corners, normal):
    """Make the 4 corner list CCW seen from the outside (front face)."""
    c = np.array(corners, np.float64)
    if np.dot(np.cross(c[1] - c[0], c[2] - c[0]), normal) < 0:
        return [0, 3, 2, 1]
    return [0, 1, 2, 3]


def build_world_mesh(world, atlas, ids, separate_ao=True):
    B = world.blocks
    mb = MeshBuilder(atlas, ids, separate_ao)
    block_ids = np.array([ids.get(b[0], -1) for b in BLOCKS], np.float32)
    layer_index = {l: i for i, l in enumerate(LAYERS)}
    Bp = np.pad(B, 1, mode="constant", constant_values=0)
    Bp[:, 0, :] = BID["stone"]  # nothing below the world
    opq_p = OPAQUE[Bp]
    sky_p = np.pad(world.sky, 1, mode="edge")
    blk_p = np.pad(world.blk, 1, mode="constant", constant_values=0)
    liquid_top = np.zeros(B.shape, bool)
    above = np.concatenate([B[:, 1:, :], np.zeros((SX, 1, SZ), B.dtype)], axis=1)
    liquid_top = (np.array([s == "liquid" for s in SHAPE])[B]) & (above != B)

    # per-block per-face texture tile
    def tile(bid, face):
        t = BLOCKS[bid][3]
        return t[0] if face == 2 else (t[2] if face == 3 else t[1])

    cube_mask = IS_CUBEISH[B]
    for fi, (n, tdir, vdir) in enumerate(DIRS):
        n = np.array(n)
        tdir = np.array(tdir)
        vdir = np.array(vdir)
        nb = Bp[1 + n[0]:1 + n[0] + SX, 1 + n[1]:1 + n[1] + SY, 1 + n[2]:1 + n[2] + SZ]
        vis = cube_mask & ~OPAQUE[nb] & ~(CULL_SAME[B] & (nb == B))
        # liquid top faces that are lowered are visible against anything non-opaque above (even same? no: above != B)
        idx = np.argwhere(vis)
        if len(idx) == 0:
            continue
        bids = B[idx[:, 0], idx[:, 1], idx[:, 2]]
        Q = len(idx)
        # corners (u,v) order (0,0),(0,1),(1,1),(1,0)
        uvs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        corner_off = []
        for (u, v) in uvs:
            corner_off.append(0.5 + n * 0.5 + (u - 0.5) * tdir + (v - 0.5) * vdir)
        order = _fix_winding(corner_off, n)
        uvs = [uvs[k] for k in order]
        corner_off = np.array([corner_off[k] for k in order])           # (4,3)
        pos = idx[:, None, :].astype(np.float64) + corner_off[None, :, :]  # (Q,4,3)
        # lowered liquid surfaces
        lt = liquid_top[idx[:, 0], idx[:, 1], idx[:, 2]]
        if lt.any():
            ytop = idx[:, 1].astype(np.float64) + 1.0
            hi = np.isclose(pos[:, :, 1], ytop[:, None]) & lt[:, None]
            pos[:, :, 1] = np.where(hi, pos[:, :, 1] - 2.0 / 16.0, pos[:, :, 1])
        # texture coordinates
        tiles = np.array([atlas.index[tile(b, fi)] for b in bids], np.float64)  # (Q,2)
        uvq = np.array(uvs, np.float64)
        uv = (tiles[:, None, :] + uvq[None, :, :]) / atlas.grid
        # smooth light + AO samples
        cell = idx + 1  # padded coords
        front = cell + n
        col_ao = np.ones((Q, 4), np.float64)
        lm = np.zeros((Q, 4, 2), np.float64)
        fb = blk_p[front[:, 0], front[:, 1], front[:, 2]].astype(np.float64)
        fs = sky_p[front[:, 0], front[:, 1], front[:, 2]].astype(np.float64)
        liquid = np.array([SHAPE[b] == "liquid" for b in bids])
        for k, (u, v) in enumerate(uvs):
            su = -1 if u == 0 else 1
            sv = -1 if v == 0 else 1
            s1 = front + su * tdir
            s2 = front + sv * vdir
            cc = front + su * tdir + sv * vdir
            o1 = opq_p[s1[:, 0], s1[:, 1], s1[:, 2]]
            o2 = opq_p[s2[:, 0], s2[:, 1], s2[:, 2]]
            oc = opq_p[cc[:, 0], cc[:, 1], cc[:, 2]] | (o1 & o2)
            # vanilla AO: shade brightness 0.2 for full opaque cubes, 1.0 otherwise
            col_ao[:, k] = (1.0 + np.where(o1, 0.2, 1.0) + np.where(o2, 0.2, 1.0) + np.where(oc, 0.2, 1.0)) / 4.0
            for ch, arr, fv in ((0, blk_p, fb), (1, sky_p, fs)):
                a1 = np.where(o1, fv, arr[s1[:, 0], s1[:, 1], s1[:, 2]])
                a2 = np.where(o2, fv, arr[s2[:, 0], s2[:, 1], s2[:, 2]])
                ac = np.where(oc, fv, arr[cc[:, 0], cc[:, 1], cc[:, 2]])
                lm[:, k, ch] = (fv + a1 + a2 + ac) / 4.0
        # liquids: flat light, no AO
        if liquid.any():
            col_ao[liquid] = 1.0
            own_b = blk_p[cell[:, 0], cell[:, 1], cell[:, 2]].astype(np.float64)
            own_s = sky_p[cell[:, 0], cell[:, 1], cell[:, 2]].astype(np.float64)
            lm[liquid, :, 0] = np.maximum(fb, own_b)[liquid, None]
            lm[liquid, :, 1] = np.maximum(fs, own_s)[liquid, None]
        # emitters are fully block-lit
        em = EMIT[bids].astype(np.float64)
        lm[:, :, 0] = np.maximum(lm[:, :, 0], em[:, None] * (em[:, None] > 0))
        lm = lm * 16.0
        # colour: tint + AO
        tint = np.ones((Q, 3))
        for b in np.unique(bids):
            tn = BLOCKS[b][4]
            if tn and (tn != "grass_top" or fi == 2):
                tint[bids == b] = TINTS[tn]
        color = np.zeros((Q, 4, 4))
        if separate_ao:
            color[:, :, :3] = tint[:, None, :]
            color[:, :, 3] = col_ao
        else:
            color[:, :, :3] = tint[:, None, :] * col_ao[:, :, None]
            color[:, :, 3] = 1.0
        normal = np.repeat(n[None, :].astype(np.float64), Q, axis=0)
        center = idx + 0.5
        layers = [LAYER[b] for b in bids]
        ent = np.stack([block_ids[bids], np.array([layer_index[l] for l in layers], np.float64)], 1)
        verts = _assemble(pos, color, uv, lm, normal, ent, center, em)
        for l in LAYERS:
            sel = np.array([x == l for x in layers])
            if sel.any():
                mb.add(l, verts.reshape(Q, 4, VERTEX_FLOATS)[sel].reshape(-1, VERTEX_FLOATS))

    # ---- special shapes (python loop: few blocks)
    special = np.argwhere(np.isin(B, [i for i, s in enumerate(SHAPE) if s in ("cross", "torch", "lantern", "slab", "pane")]))
    for (x, y, z) in special:
        b = int(B[x, y, z])
        name, shape = BLOCKS[b][0], SHAPE[b]
        bl, sk = float(world.blk[x, y, z]), float(world.sky[x, y, z])
        if shape == "slab":
            sk = float(world.sky[x, min(y + 1, SY - 1), z])
            bl = float(world.blk[x, min(y + 1, SY - 1), z])
        lmv = (max(bl, EMIT[b]) * 16.0, sk * 16.0)
        tint = TINTS.get(BLOCKS[b][4], (1, 1, 1)) if BLOCKS[b][4] else (1, 1, 1)
        quads = []
        if shape == "cross":
            for (a, c) in (((0.15, 0.15), (0.85, 0.85)), ((0.15, 0.85), (0.85, 0.15))):
                p0 = np.array([x + a[0], y + 1.0, z + a[1]])
                p1 = np.array([x + c[0], y + 1.0, z + c[1]])
                q = [p0, p0 - [0, 1, 0], p1 - [0, 1, 0], p1]
                uvq = [(0, 0), (0, 1), (1, 1), (1, 0)]
                nrm = np.cross(q[1] - q[0], q[2] - q[0])
                nrm = nrm / np.linalg.norm(nrm)
                quads.append((q, uvq, nrm, BLOCKS[b][3][1]))
                quads.append((q[::-1], uvq[::-1], -nrm, BLOCKS[b][3][1]))
        elif shape == "slab":
            quads += _box_quads(x, y, z, (0, 0, 0), (1, 0.5, 1), BLOCKS[b][3], B)
        elif shape == "torch":
            quads += _box_quads(x, y, z, (7 / 16, 0, 7 / 16), (9 / 16, 10 / 16, 9 / 16), BLOCKS[b][3], None)
        elif shape == "lantern":
            quads += _box_quads(x, y, z, (5 / 16, 0, 5 / 16), (11 / 16, 9 / 16, 11 / 16), BLOCKS[b][3], None)
        elif shape == "pane":
            # thin 2/16 pane (centre post only, no side arms); axis "x" spans x (thin in z), "z" spans z.
            # Faces against a neighbouring pane or an opaque block are dropped (vanilla connects panes).
            ax = world.pane_axis.get((int(x), int(y), int(z)), "x")
            lo, hi = ((0, 0, 7 / 16), (1, 1, 9 / 16)) if ax == "x" else ((7 / 16, 0, 0), (9 / 16, 1, 1))
            for qd in _box_quads(x, y, z, lo, hi, BLOCKS[b][3], B):
                nn = qd[2]
                q = np.array([x, y, z]) + nn.astype(int)
                along = (ax == "x" and abs(nn[0]) > 0.5) or (ax == "z" and abs(nn[2]) > 0.5)
                if along and 0 <= q[0] < SX and 0 <= q[1] < SY and 0 <= q[2] < SZ and \
                        (SHAPE[B[q[0], q[1], q[2]]] == "pane" or OPAQUE[B[q[0], q[1], q[2]]]):
                    continue
                if abs(nn[1]) > 0.5 and 0 <= q[1] < SY and SHAPE[B[x, q[1], z]] == "pane":
                    continue
                quads.append(qd)
        if not quads:
            continue
        Q = len(quads)
        pos = np.array([q[0] for q in quads], np.float64)
        uvl = np.array([q[1] for q in quads], np.float64)
        tiles = np.array([atlas.index[q[3]] for q in quads], np.float64)
        uv = (tiles[:, None, :] + uvl) / atlas.grid
        nrm = np.array([q[2] for q in quads], np.float64)
        color = np.ones((Q, 4, 4))
        color[:, :, :3] = tint
        lm = np.tile(np.array(lmv), (Q, 4, 1))
        ent = np.tile(np.array([block_ids[b], layer_index[LAYER[b]]], np.float64), (Q, 1))
        center = np.tile(np.array([x + 0.5, y + 0.5, z + 0.5]), (Q, 1))
        em = np.full(Q, float(EMIT[b]))
        mb.add(LAYER[b], _assemble(pos, color, uv, lm, nrm, ent, center, em))
    return mb.finish()


def _box_quads(x, y, z, lo, hi, texs, B):
    """Axis-aligned box inside block (x,y,z); UVs use the matching sub-rectangle of the texture."""
    lo = np.array(lo, np.float64)
    hi = np.array(hi, np.float64)
    out = []
    for fi, (n, tdir, vdir) in enumerate(DIRS):
        n = np.array(n)
        tdir = np.array(tdir, np.float64)
        vdir = np.array(vdir, np.float64)
        # cull full-size boundary faces against opaque neighbours
        if B is not None:
            on_boundary = (n > 0) & np.isclose(hi, 1.0) | (n < 0) & np.isclose(lo, 0.0)
            if on_boundary[np.nonzero(n)[0][0]]:
                q = np.array([x, y, z]) + n
                if 0 <= q[0] < SX and 0 <= q[1] < SY and 0 <= q[2] < SZ and OPAQUE[B[q[0], q[1], q[2]]]:
                    continue
        center = (lo + hi) / 2
        ext = (hi - lo) / 2
        fc = center + n * ext
        corners, uvs = [], []
        for (u, v) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            p = fc + (u - 0.5) * 2 * tdir * ext + (v - 0.5) * 2 * vdir * ext
            corners.append(p)
            # texture sub-rect: project block-local position onto t / v axes
            tu = np.dot(p, tdir) if tdir.sum() > 0 else 1.0 + np.dot(p, tdir)
            tv = np.dot(p, vdir) if vdir.sum() > 0 else 1.0 + np.dot(p, vdir)
            uvs.append((float(np.clip(tu, 0, 1)), float(np.clip(tv, 0, 1))))
        order = _fix_winding(corners, n)
        corners = [corners[k] + np.array([x, y, z]) for k in order]
        uvs = [uvs[k] for k in order]
        tex = texs[0] if fi == 2 else (texs[2] if fi == 3 else texs[1])
        out.append((corners, uvs, n.astype(np.float64), tex))
    return out


def build_entity_mesh(atlas, entity_block_id=-1, light=(0, 15)):
    """A pig-like entity in model space (feet at origin): body + head boxes."""
    quads = []
    for lo, hi in (((-0.45, 0.35, -0.6), (0.45, 1.0, 0.6)), ((-0.3, 0.55, 0.6), (0.3, 1.1, 1.05)),
                   ((-0.4, 0.0, -0.5), (-0.15, 0.35, -0.25)), ((0.15, 0.0, -0.5), (0.4, 0.35, -0.25)),
                   ((-0.4, 0.0, 0.25), (-0.15, 0.35, 0.5)), ((0.15, 0.0, 0.25), (0.4, 0.35, 0.5))):
        lo = np.array(lo)
        hi = np.array(hi)
        for fi, (n, tdir, vdir) in enumerate(DIRS):
            n = np.array(n, np.float64)
            tdir = np.array(tdir, np.float64)
            vdir = np.array(vdir, np.float64)
            c = (lo + hi) / 2
            e = (hi - lo) / 2
            fc = c + n * e
            corners = [fc + (u - 0.5) * 2 * tdir * e + (v - 0.5) * 2 * vdir * e for (u, v) in [(0, 0), (0, 1), (1, 1), (1, 0)]]
            uvs = [(0, 0), (0, 1), (1, 1), (1, 0)]
            order = _fix_winding(corners, n)
            quads.append(([corners[k] for k in order], [uvs[k] for k in order], n, "entity"))
    Q = len(quads)
    pos = np.array([q[0] for q in quads])
    tiles = np.array([atlas.index[q[3]] for q in quads], np.float64)
    uv = (tiles[:, None, :] + np.array([q[1] for q in quads], np.float64)) / atlas.grid
    nrm = np.array([q[2] for q in quads])
    color = np.ones((Q, 4, 4))
    lm = np.tile(np.array([light[0] * 16.0, light[1] * 16.0]), (Q, 4, 1))
    ent = np.tile(np.array([entity_block_id, 0.0]), (Q, 1))
    center = pos.mean(axis=1)
    return _assemble(pos, color, uv, lm, nrm, ent, center, np.zeros(Q))


def build_sky_mesh(radius=200.0, rings=16, segs=32):
    """Sphere around the camera (gbuffers_skybasic)."""
    quads = []
    for i in range(rings):
        t0 = np.pi * i / rings
        t1 = np.pi * (i + 1) / rings
        for j in range(segs):
            p0 = 2 * np.pi * j / segs
            p1 = 2 * np.pi * (j + 1) / segs

            def sp(t, p):
                return np.array([np.sin(t) * np.cos(p), np.cos(t), np.sin(t) * np.sin(p)]) * radius
            q = [sp(t0, p0), sp(t1, p0), sp(t1, p1), sp(t0, p1)]
            nrm = -(q[0] + q[2]) / 2
            nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
            quads.append(q)
    Q = len(quads)
    pos = np.array(quads)
    uv = np.zeros((Q, 4, 2))
    nrm = -pos.mean(axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    color = np.ones((Q, 4, 4))
    lm = np.full((Q, 4, 2), 240.0)
    return _assemble(pos, color, uv, lm, nrm, np.full((Q, 2), -1.0), np.zeros((Q, 3)), np.zeros(Q))


def build_rain_mesh(world, cam_idx, atlas, radius=6, t=0.0):
    """Vertical rain quads on sky-exposed columns around the camera (block index coords)."""
    quads = []
    cx, cy, cz = (int(np.floor(v)) for v in cam_idx)
    for x in range(cx - radius, cx + radius + 1):
        for z in range(cz - radius, cz + radius + 1):
            if (x - cx) ** 2 + (z - cz) ** 2 > radius * radius or not (0 <= x < SX and 0 <= z < SZ):
                continue
            col = world.blocks[x, :, z]
            ground = int(np.nonzero(col)[0].max()) + 1 if col.any() else 0
            y0 = max(ground, cy - radius)
            y1 = cy + radius
            if y1 <= y0:
                continue
            # one quad per block of height so the streak texture keeps its 16 px/block scale (vanilla tiles it)
            for (a, b) in (((x, z + 0.5), (x + 1, z + 0.5)), ((x + 0.5, z), (x + 0.5, z + 1))):
                for yy in range(y0, y1):
                    q = [np.array([a[0], yy + 1, a[1]]), np.array([a[0], yy, a[1]]),
                         np.array([b[0], yy, b[1]]), np.array([b[0], yy + 1, b[1]])]
                    uvl = [(0, 0), (0, 1), (1, 1), (1, 0)]
                    quads.append((q, uvl))   # drawn with culling off: one quad is already double-sided
    if not quads:
        return np.zeros((0, VERTEX_FLOATS), np.float32)
    Q = len(quads)
    pos = np.array([q[0] for q in quads], np.float64)
    tx, ty = atlas.index["rain"]
    uvl = np.array([q[1] for q in quads], np.float64)
    uv = np.stack([(tx + uvl[..., 0]) / atlas.grid, (ty + uvl[..., 1]) / atlas.grid], -1)
    nrm = np.tile(np.array([0.0, 0.0, 1.0]), (Q, 1))
    color = np.ones((Q, 4, 4))
    lm = np.tile(np.array([0.0, 240.0]), (Q, 4, 1))
    return _assemble(pos, color, uv, lm, nrm, np.full((Q, 2), -1.0), pos.mean(axis=1), np.zeros(Q))


# ---------------------------------------------------------------- Distant Horizons LOD ring
# Iris StandardMacros DH_BLOCK_* values (EDhApiBlockMaterial)
DH_UNKNOWN, DH_LEAVES, DH_STONE, DH_WOOD, DH_METAL, DH_DIRT, DH_LAVA, DH_DEEPSLATE, DH_SNOW, DH_SAND, \
    DH_TERRACOTTA, DH_NETHER_STONE, DH_WATER, DH_GRASS, DH_AIR, DH_ILLUMINATED = range(16)
# LOD block colours as DH bakes them (sRGB texture average x biome tint)
DH_COLORS = {
    DH_GRASS: (0.47, 0.65, 0.30), DH_DIRT: (0.53, 0.38, 0.26), DH_STONE: (0.49, 0.49, 0.50),
    DH_SNOW: (0.94, 0.97, 0.98), DH_SAND: (0.86, 0.81, 0.62), DH_LEAVES: (0.28, 0.46, 0.16),
    DH_ILLUMINATED: (0.98, 0.84, 0.52), DH_LAVA: (0.90, 0.42, 0.08), DH_WOOD: (0.43, 0.34, 0.21),
    DH_NETHER_STONE: (0.44, 0.20, 0.19),
}
DH_WATER_COLOR = (0.25, 0.42, 0.85, 0.75)
# DH reports end stone as DH_STONE (EDhApiBlockMaterial has no end-stone class), but bakes the LOD colour
# from the end_stone texture: the sRGB mean of textures.py's end_stone, so End LODs match the vanilla end
# stone next to them (the overworld DH_STONE grey made a brightness step at the boundary)
END_STONE_LOD_COLOR = (0.81, 0.82, 0.59)


def _vnoise(x, z, seed):
    """Smooth value noise in [0,1] on unit cells (vectorised)."""
    xi, zi = np.floor(x), np.floor(z)
    fx, fz = x - xi, z - zi
    ux, uz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)

    def h(a, b):
        v = (a.astype(np.int64) * 374761393 + b.astype(np.int64) * 668265263 + seed * 2654435761) & 0xFFFFFFFF
        v = (v ^ (v >> 13)) * 1274126177 & 0xFFFFFFFF
        return ((v ^ (v >> 16)) & 0xFFFF) / 65535.0
    a, b, c, d = h(xi, zi), h(xi + 1, zi), h(xi, zi + 1), h(xi + 1, zi + 1)
    return (a * (1 - ux) + b * ux) * (1 - uz) + (c * (1 - ux) + d * ux) * uz


def dh_height(x, z, dimension="overworld"):
    """LOD terrain height (block-index y, float) at index coordinates x, z (vectorised). Continues the
    scene's own height function at its border; hills, a lake / sea, a mountain range and a forest farther
    out (world-relative so the ring is the same every frame)."""
    base = (33.0 + 1.6 * np.sin(x * 0.13) * np.cos(z * 0.11) + 1.2 * np.sin((x + 2 * z) * 0.05)
            + 0.8 * np.cos(x * 0.31 + 1.0) * np.sin(z * 0.27))
    if dimension != "overworld":
        return base + 10.0 * _vnoise(x / 90.0, z / 90.0, 7)
    cx, cz = x - 48.0, z - 48.0
    d = np.hypot(cx, cz)
    ramp = np.clip((d - 60.0) / 120.0, 0.0, 1.0)
    hills = 14.0 * (_vnoise(x / 140.0, z / 140.0, 3) + 0.5 * _vnoise(x / 60.0, z / 60.0, 4) - 0.6) * ramp
    # mountain range across the north-east (the preset cameras look that way), snow-capped peaks
    ang = np.arctan2(cz, cx)
    sector = np.exp(-((ang + 1.05) / 0.9) ** 2)
    ridge = 1.0 - np.abs(2.0 * _vnoise(x / 110.0, z / 110.0, 5) - 1.0)
    mountains = 95.0 * np.exp(-((d - 620.0) / 230.0) ** 2) * sector * (0.35 + 0.65 * ridge)
    # a large lake east of the scene
    lake = 22.0 * np.clip(1.0 - np.hypot(cx - 300.0, cz + 60.0) / 190.0, 0.0, 1.0) ** 0.6
    return base + hills + mountains - lake


def build_dh_meshes(dimension="overworld", radius=1024, cell=16):
    """Coarse LOD terrain like Distant Horizons draws it: cell x cell columns over a square of the given
    radius around the scene (none inside the scene's own 96x96 footprint, where vanilla chunks are), top
    quads plus side quads down to lower neighbours, dhMaterialId in mc_Entity.x (iris_DhMaterial), the
    LOD colour in gl_Color, lightmap ((block + 0.5) / 16, (sky + 0.5) / 16) for an identity texture
    matrix (DHTerrainTransformer), axis-aligned normals; water tops (at the vanilla water surface height)
    go to a separate translucent mesh for dh_water. Positions are scene block-index coordinates (relative
    to WORLD_ORIGIN, like the terrain mesh). Returns {"dh_opaque": verts, "dh_water": verts}."""
    n = int(2 * radius // cell)
    lo = 48 - radius
    xs = lo + np.arange(n) * cell
    X0, Z0 = np.meshgrid(xs, xs, indexing="ij")
    Xc, Zc = X0 + cell * 0.5, Z0 + cell * 0.5
    H = np.floor(dh_height(Xc, Zc, dimension))
    sea_top = SEA + 14.0 / 16.0
    water = (H < SEA) & (dimension == "overworld")
    snow = H > 95.0
    stone = (H > 62.0) & ~snow
    sand = ~water & (H <= SEA + 2)
    forest = (_vnoise(Xc / 70.0, Zc / 70.0, 11) > 0.62) & ~water & ~stone & ~snow & ~sand
    H = np.where(forest, H + 6.0, H)            # canopy
    top = np.full(H.shape, DH_GRASS)
    top = np.where(forest, DH_LEAVES, top)
    top = np.where(sand, DH_SAND, top)
    top = np.where(water, DH_SAND, top)         # lake floor
    top = np.where(stone, DH_STONE, top)
    top = np.where(snow, DH_SNOW, top)
    # a village (glowing at night) on the near shore and a lava pool in the mountains
    village = (np.abs(Xc - 190.0) < 40) & (np.abs(Zc + 70.0) < 40) & ~water & (((X0 // cell + Z0 // cell) % 3) == 0)
    top = np.where(village, DH_ILLUMINATED, top)
    lava = (np.abs(Xc - 430.0) < 24) & (np.abs(Zc + 520.0) < 24)
    top = np.where(lava, DH_LAVA, top)
    if dimension == "nether":
        top = np.where(lava, DH_LAVA, DH_NETHER_STONE)
    elif dimension == "end":
        top = np.full(H.shape, DH_STONE)
    inside = (X0 >= 0) & (X0 + cell <= SX) & (Z0 >= 0) & (Z0 + cell <= SZ)
    keep = ~inside

    quads = {"opaque": [], "water": []}   # (corners (Q,4,3), normal (Q,3), mat (Q,), color (Q,4), block (Q,))

    def add(kind, corners, normal, mat, color, block):
        if len(corners):
            quads[kind].append((corners, normal, mat, color, block))

    palette = dict(DH_COLORS)
    if dimension == "end":
        palette[DH_STONE] = END_STONE_LOD_COLOR   # material id stays DH_STONE

    def colors_of(mat, shade=1.0):
        return np.array([palette.get(int(m), (0.5, 0.5, 0.5)) + (1.0,) for m in mat]) * np.array([shade, shade, shade, 1.0])

    sel = np.nonzero(keep)
    x0, z0, h = X0[sel].astype(np.float64), Z0[sel].astype(np.float64), H[sel]
    x1, z1 = x0 + cell, z0 + cell
    m = top[sel]
    c = np.stack([np.stack([x0, h, z0], -1), np.stack([x0, h, z1], -1), np.stack([x1, h, z1], -1),
                  np.stack([x1, h, z0], -1)], 1)
    add("opaque", c, np.tile([0.0, 1.0, 0.0], (len(m), 1)), m, colors_of(m),
        np.where(m == DH_ILLUMINATED, 15, 0))
    wsel = np.nonzero(keep & water)
    wx0, wz0 = X0[wsel].astype(np.float64), Z0[wsel].astype(np.float64)
    wx1, wz1 = wx0 + cell, wz0 + cell
    wh = np.full(wx0.shape, sea_top)
    wc = np.stack([np.stack([wx0, wh, wz0], -1), np.stack([wx0, wh, wz1], -1), np.stack([wx1, wh, wz1], -1),
                   np.stack([wx1, wh, wz0], -1)], 1)
    add("water", wc, np.tile([0.0, 1.0, 0.0], (len(wx0), 1)), np.full(len(wx0), DH_WATER),
        np.tile(DH_WATER_COLOR, (len(wx0), 1)), np.zeros(len(wx0), int))
    # sides towards lower neighbours (the cell's own side material: dirt, stone in the mountains)
    side_mat = np.where(stone | snow | (H > 62.0), DH_STONE, DH_DIRT)
    side_mat = np.where(forest, DH_LEAVES, side_mat)
    if dimension == "nether":
        side_mat = np.full(H.shape, DH_NETHER_STONE)
    elif dimension == "end":
        side_mat = np.full(H.shape, DH_STONE)
    for axis, sgn in ((0, 1), (0, -1), (1, 1), (1, -1)):
        Hn = np.roll(H, -sgn, axis=axis)
        valid = np.ones(H.shape, bool)
        if axis == 0:
            valid[-1 if sgn > 0 else 0, :] = False
        else:
            valid[:, -1 if sgn > 0 else 0] = False
        s2 = np.nonzero(keep & valid & (Hn < H))
        if not len(s2[0]):
            continue
        a0, b0 = X0[s2].astype(np.float64), Z0[s2].astype(np.float64)
        hh, hn = H[s2], Hn[s2]
        if axis == 0 and sgn > 0:      # +x face at x1
            xa = a0 + cell
            c = np.stack([np.stack([xa, hn, b0], -1), np.stack([xa, hh, b0], -1), np.stack([xa, hh, b0 + cell], -1),
                          np.stack([xa, hn, b0 + cell], -1)], 1)
            nrm = [1.0, 0.0, 0.0]
        elif axis == 0:                # -x face at x0
            c = np.stack([np.stack([a0, hn, b0 + cell], -1), np.stack([a0, hh, b0 + cell], -1),
                          np.stack([a0, hh, b0], -1), np.stack([a0, hn, b0], -1)], 1)
            nrm = [-1.0, 0.0, 0.0]
        elif sgn > 0:                  # +z face at z1
            za = b0 + cell
            c = np.stack([np.stack([a0 + cell, hn, za], -1), np.stack([a0 + cell, hh, za], -1),
                          np.stack([a0, hh, za], -1), np.stack([a0, hn, za], -1)], 1)
            nrm = [0.0, 0.0, 1.0]
        else:                          # -z face at z0
            c = np.stack([np.stack([a0, hn, b0], -1), np.stack([a0, hh, b0], -1), np.stack([a0 + cell, hh, b0], -1),
                          np.stack([a0 + cell, hn, b0], -1)], 1)
            nrm = [0.0, 0.0, -1.0]
        sm = side_mat[s2]
        add("opaque", c, np.tile(nrm, (len(sm), 1)), sm, colors_of(sm), np.zeros(len(sm), int))

    out = {}
    for kind, parts in quads.items():
        if not parts:
            out["dh_" + kind] = np.zeros((0, VERTEX_FLOATS), np.float32)
            continue
        pos = np.concatenate([p[0] for p in parts])
        nrm = np.concatenate([p[1] for p in parts])
        mat = np.concatenate([p[2] for p in parts]).astype(np.float64)
        col = np.concatenate([p[3] for p in parts])
        blk = np.concatenate([p[4] for p in parts]).astype(np.float64)
        # front faces must be CCW seen from outside (back-face culling like the opaque LOD pass)
        e = np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0])
        assert np.all(np.sum(e * nrm, axis=1) > 0), "DH LOD quad winding"
        Q = len(pos)
        lm = np.stack([(blk + 0.5) / 16.0, np.full(Q, 15.5 / 16.0)], -1)[:, None, :].repeat(4, 1)
        color = col[:, None, :].repeat(4, 1)
        out["dh_" + kind] = _assemble(pos, color, np.zeros((Q, 4, 2)), lm, nrm, np.stack([mat, np.zeros(Q)], -1),
                                      pos.mean(axis=1), np.zeros(Q))
    return out
