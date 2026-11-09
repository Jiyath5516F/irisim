"""irisim — a headless mini-Iris for Minecraft Java shader packs.

Runs a pack's real programs (shaders/world*/) in Iris order on Mesa llvmpipe through an OpenGL 4.5
compatibility context, against a synthetic Minecraft-like voxel scene, and writes the final image plus
per-pass buffer dumps, timings and NaN/Inf statistics.

  irisim path/to/MyPack --scenario noon
  irisim path/to/MyPack.zip --scenario all --width 640 --height 360 --frames 4
  irisim path/to/MyPack --scenario cave --frames 4 --set SHADOW_QUALITY=2 --off BLOOM
  irisim path/to/MyPack --time 13000 --camera overview --rain 1 --name stormy
  irisim path/to/MyPack --scenario noon --dh --frames 12 --name dh_noon    # Distant Horizons LOD ring
  irisim path/to/MyPack --list-options                                      # options the pack declares

See README.md and docs/HOW_IT_WORKS.md.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

from . import __version__
from . import camera as cam
from . import options as optmod
from . import pack
from . import scene
from . import textures

HERE = os.path.dirname(os.path.abspath(__file__))
O = scene.WORLD_ORIGIN.astype(np.float64)

# Camera and scenario presets: built-in ones from presets/default.json, more from --presets FILE.
CAMERAS = {}
SCENARIOS = {}
ALL_SCENARIOS = []
STRUCTURES = ("glass_gallery", "village", "gallery", "poles", "hall", "furnace")


def load_presets(path):
    """Merge a preset JSON file (cameras / scenarios / all) into CAMERAS / SCENARIOS / ALL_SCENARIOS."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for name, c in data.get("cameras", {}).items():
        eye, tgt = c["eye"], c["target"]
        CAMERAS[name] = ((float(eye[0]), None if eye[1] is None else float(eye[1]), float(eye[2])),
                         tuple(float(v) for v in tgt))
    for name, sc in data.get("scenarios", {}).items():
        sc = dict(sc)
        for s in sc.pop("structures", []) or []:
            if s not in STRUCTURES:
                raise ValueError(f"{path}: scenario {name}: unknown structure {s!r} (have {', '.join(STRUCTURES)})")
            sc[s] = True
        SCENARIOS[name] = sc
    if "all" in data:
        ALL_SCENARIOS[:] = list(data["all"])


load_presets(os.path.join(HERE, "presets", "default.json"))

# test-structure defaults
GALLERY_KINDS = "torch,soul_torch,redstone_torch,lantern,soul_lantern,sea_lantern,glowstone,redstone_lamp,shroomlight"
POLE_KINDS = "lantern,torch,glowstone,sea_lantern,soul_lantern"
DIMENSIONS = {"world0": "overworld", "world-1": "nether", "world1": "end"}
# per-frame log line: these uniforms when the pack declares them as custom uniforms (+ --log-uniforms)
LOG_UNIFORMS = ("endFlashIntensity",)


def find_shaders_dir(path, tmp_holder):
    """PACK argument -> the pack's shaders/ directory. Accepts the shaders/ directory itself, a pack root
    containing shaders/, or a .zip pack (extracted to a temporary directory)."""
    path = os.path.abspath(path)
    if os.path.isfile(path) and zipfile.is_zipfile(path):
        tmp = tempfile.mkdtemp(prefix="irisim_pack_")
        tmp_holder.append(tmp)
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        path = tmp
    if not os.path.isdir(path):
        raise SystemExit(f"irisim: {path}: not a directory or .zip shader pack")
    if os.path.isdir(os.path.join(path, "shaders")):
        return os.path.join(path, "shaders")
    if any(os.path.exists(os.path.join(path, n)) for n in ("shaders.properties", "world0", "final.fsh",
                                                            "composite.fsh", "gbuffers_basic.fsh")):
        return path
    # a zip with one top-level folder, or a pack nested one level down
    for d in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, d, "shaders")):
            return os.path.join(path, d, "shaders")
    raise SystemExit(f"irisim: no shaders/ directory found in {path}")


def parse_args(argv):
    # --presets first, so that --scenario / --camera choices include the loaded presets
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--presets", action="append", default=[])
    known, _ = pre.parse_known_args(argv)
    for p in known.presets:
        load_presets(p)
    ap = argparse.ArgumentParser(prog="irisim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", nargs="?", help="shader pack: its root folder, its shaders/ folder, or a .zip")
    ap.add_argument("--version", action="version", version=f"irisim {__version__}")
    ap.add_argument("--shaders", help="alternative to PACK: the shaders/ directory to load")
    ap.add_argument("--presets", action="append", default=[], metavar="FILE",
                    help="load more camera / scenario presets from a JSON file (repeatable; see examples/presets)")
    ap.add_argument("--scenario", help="preset (time, camera, weather, world); 'all' renders the default set "
                                       "one after another. --list-scenarios shows them")
    ap.add_argument("--list-scenarios", action="store_true", help="print scenarios and cameras, then exit")
    ap.add_argument("--list-options", action="store_true",
                    help="print the options the pack declares (as Iris discovers them), then exit")
    ap.add_argument("--name", help="output folder name under --out (default: scenario or 'custom')")
    ap.add_argument("--out", help="output root directory (default ./irisim-out)")
    ap.add_argument("--xvfb", choices=["auto", "always", "never"], default="auto",
                    help="run under xvfb-run: auto = only when DISPLAY is not set (default)")
    ap.add_argument("--world", choices=["world0", "world-1", "world1"])
    ap.add_argument("--time", type=int, help="worldTime ticks (6000 noon, 12500 sunset, 18000 midnight)")
    ap.add_argument("--time-speed", type=float, default=0.0, help="worldTime ticks advanced per frame")
    ap.add_argument("--rain", type=float, help="rainStrength 0..1 (also wetness)")
    ap.add_argument("--thunder", type=float, help="thunderStrength 0..1")
    ap.add_argument("--rain-ramp", metavar="START,FRAMES",
                    help="ramp rainStrength (and wetness unless --wetness) linearly from 0 at frame START to --rain "
                         "FRAMES frames later (vanilla ramps 0.01 per tick, i.e. 0 -> 1 in 5 s)")
    ap.add_argument("--wetness", type=float, help="override the wetness uniform (default: equal to rainStrength)")
    ap.add_argument("--precipitation", choices=["none", "rain", "snow"],
                    help="biome_precipitation 0 / 1 / 2 (default: rain in the Overworld, none elsewhere); 'none' "
                         "also hides the weather quads, as vanilla does in dry biomes")
    ap.add_argument("--temperature", type=float, default=0.8, help="biome temperature uniform (vanilla -0.7 .. 2.0)")
    ap.add_argument("--rainfall", type=float, default=0.4, help="biome rainfall (downfall) uniform 0..1")
    ap.add_argument("--fog-color", metavar="R,G,B",
                    help="override the fogColor uniform (vanilla sRGB 0..1, e.g. Nether biomes: wastes 0.2,0.031,0.031, "
                         "crimson 0.2,0.012,0.012, warped 0.102,0.020,0.102, soul sand valley 0.106,0.278,0.271, "
                         "basalt deltas 0.408,0.373,0.439)")
    ap.add_argument("--end-flash", type=float, default=0.0,
                    help="endFlashIntensity 0..1 in the End (Iris, MC 1.21.9+; 0 elsewhere)")
    ap.add_argument("--end-flash-dir", metavar="AZ,EL", default="60,25",
                    help="direction of the End flash (endFlashPosition, End only): azimuth in degrees clockwise from "
                         "north (-Z) towards east (+X), elevation in degrees (default 60,25)")
    ap.add_argument("--t0", type=float, default=0.0,
                    help="frameTimeCounter at frame 0 in s: frameTimeCounter = (t0 + frame * dt) mod 3600")
    ap.add_argument("--dt", type=float, default=1.0 / 60.0, help="frameTime in s (default 1/60)")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--camera", help="camera preset (--list-scenarios shows them)")
    ap.add_argument("--pos", help="eye position x,y,z in scene block-index coords (0..96, 0..48, 0..96)")
    ap.add_argument("--yaw", type=float, help="degrees; 0 looks towards +z, -90 towards +x")
    ap.add_argument("--pitch", type=float, help="degrees; positive looks down")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--motion", type=float, default=0.0, help="camera forward motion in blocks per frame")
    ap.add_argument("--turn", type=float, default=0.0, help="camera yaw change in degrees per frame")
    ap.add_argument("--underwater", action="store_true", help="force isEyeInWater=1")
    ap.add_argument("--vertex-space", choices=["sodium", "chunk"], default="sodium",
                    help="sodium (Iris 1.21 default): gl_Vertex camera-relative, gl_ModelViewMatrix = view rotation; "
                         "chunk: gl_Vertex region-relative, camera translation in gl_ModelViewMatrix")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--render-distance", type=int, default=None,
                    help="chunks (far = 16*chunks); default 8, with --dh 4 (vanilla chunks then end where the "
                         "scene does, so the LODs start right behind it)")
    ap.add_argument("--dh", action="store_true",
                    help="emulate Distant Horizons: DISTANT_HORIZONS + DH uniforms, an LOD terrain ring drawn with "
                         "dh_terrain (after the opaque terrain, own depth buffer) and LOD water with dh_water")
    ap.add_argument("--dh-distance", type=int, default=1024, help="DH LOD render distance in blocks (dhRenderDistance)")
    ap.add_argument("--dh-order", choices=["after", "before"], default="after",
                    help="draw the DH opaque pass after the vanilla terrain and entities (default) or before the "
                         "vanilla terrain")
    ap.add_argument("--shadow-res", type=int, default=None,
                    help="shadow map size (default 1024, applied through the shadowMapResolution option when the "
                         "pack declares it, so shader constants match; 0 = keep the pack/--set value)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="option override (repeatable)")
    ap.add_argument("--off", action="append", default=[], metavar="NAME", help="disable boolean option")
    ap.add_argument("--on", action="append", default=[], metavar="NAME", help="enable boolean option")
    ap.add_argument("--define", action="append", default=[], metavar="NAME[=VALUE]",
                    help="extra preprocessor macro for every program and the .properties files (repeatable)")
    ap.add_argument("--mc-version", type=int, help=f"MC_VERSION macro (default {optmod.HOST['mc_version']})")
    ap.add_argument("--gl-vendor", choices=["NVIDIA", "AMD", "INTEL", "MESA", "ATI", "OTHER"],
                    help=f"MC_GL_VENDOR_* macro (default {optmod.HOST['vendor']}); the renderer macro follows")
    ap.add_argument("--os", choices=["LINUX", "WINDOWS", "MAC", "UNKNOWN"],
                    help=f"MC_OS_* macro (default {optmod.HOST['os']})")
    ap.add_argument("--pbr", action="store_true", help="generate labPBR normal/specular atlases (default: flat Iris defaults)")
    ap.add_argument("--builtin-ids", action="store_true",
                    help="use irisim's built-in block id table when shaders/block.properties is missing")
    ap.add_argument("--held", help="held item name (item.properties id + light value, e.g. torch)")
    ap.add_argument("--no-entity", action="store_true")
    ap.add_argument("--test-step", action="store_true",
                    help="add a 1-block-high stone step in the cave chamber (hand-light test; e.g. --scenario cave "
                         "--held torch --pos 69.3,34.27,85.5 --yaw -90 --pitch 12)")
    tg = ap.add_argument_group(
        "test structures",
        "Optional additions to the Overworld scene (ignored elsewhere). --gallery / --poles and --hall / --furnace "
        "share an area and exclude each other. Matching cameras and scenarios: examples/presets/test-structures.json")
    tg.add_argument("--glass-gallery", action="store_true",
                    help="glass test pieces around the stained-glass wall (clear / red / blue panes, a thick clear "
                         "glass cluster, white / orange / tinted glass, ice, honey, slime) and pane / stained "
                         "windows in the house")
    tg.add_argument("--village", action="store_true",
                    help="8 lamp posts (2-block oak posts with lanterns or torches) around the house")
    tg.add_argument("--gallery", action="store_true",
                    help="coloured-light gallery at x 0..41, z 0..29: up to 9 white-concrete booths open towards -z, "
                         "one emitter on each floor")
    tg.add_argument("--gallery-kinds", default=GALLERY_KINDS, metavar="B[,B...]",
                    help=f"emitter per booth, west to east (default {GALLERY_KINDS})")
    tg.add_argument("--poles", action="store_true",
                    help="up to 5 lights on 5-block oak poles on open grass at x 0..41, z 0..29 (seen against the sky)")
    tg.add_argument("--pole-kinds", default=POLE_KINDS, metavar="B[,B...]",
                    help=f"light per pole, west to east (default {POLE_KINDS})")
    tg.add_argument("--hall", action="store_true",
                    help="closed hall (interior 30 x 7 x 30, floating at x 60..91, y 38..46, z 2..33) lit by 4 "
                         "emitters on the floor")
    tg.add_argument("--hall-wall", default="stone_bricks", metavar="B", help="hall wall block (default stone_bricks)")
    tg.add_argument("--hall-light", default="torch", metavar="B", help="hall emitter block (default torch)")
    tg.add_argument("--furnace", action="store_true",
                    help="closed room (interior 7 x 5 x 7, floating at x 80..88, y 38..44, z 4..12) with one "
                         "emitter in the middle of the floor (GI energy test)")
    tg.add_argument("--furnace-wall", default="white_concrete", metavar="B",
                    help="room wall block (default white_concrete)")
    tg.add_argument("--furnace-light", default="torch", metavar="B", help="room emitter block (default torch)")
    tg.add_argument("--origin-dy", type=int, default=0, metavar="N",
                    help="move the whole scene N blocks up in world space (negative: down); the lake surface "
                         "becomes world y 62+N (e.g. -40 puts the cave deep underground)")
    ap.add_argument("--save-frames", action="store_true", help="write frameNNN.png for every frame (without --dump all)")
    ap.add_argument("--dump", choices=["last", "all", "none"], default="last", help="per-pass dumps for which frames")
    ap.add_argument("--npy", action="append", default=[], metavar="NAME[,NAME]",
                    help="after the last frame, save the raw contents (float32 RGBA, GL row order: row 0 = bottom, "
                         "no flip; integer formats unconverted) of colortexN (the version the next reader sees), a "
                         "custom image (image or sampler name), a custom texture, depthtexN, shadowtexN or "
                         "shadowcolorN to <out>/<name>/<NAME>.npy (repeatable)")
    ap.add_argument("--npy-seq", metavar="PASS_colortexN[,...]",
                    help="per-frame float dumps: after each listed program writes the buffer, save it to "
                         "<npy-seq-dir>/<PASS>_colortex<N>_fNNN.npy (float32 RGBA, GL row order, NaN/Inf zeroed, "
                         "clamped to +-65000), e.g. deferred1_colortex0,composite_colortex0; every frame from "
                         "--npy-seq-from on also saves final_fNNN.npy (uint8 RGB as displayed, top row first) and "
                         "global_fNNN.npy (the first 256 bytes of SSBO 0 as float32, when the pack has one)")
    ap.add_argument("--npy-seq-dir", metavar="DIR", help="output folder of --npy-seq (default <out>/<name>/seq)")
    ap.add_argument("--npy-seq-from", type=int, default=0, metavar="F", help="first frame --npy-seq saves (default 0)")
    ap.add_argument("--log-uniforms", default="", metavar="NAME[,NAME]",
                    help="log the value of these uniforms (built-in or custom uniform./variable. ones) every frame")
    ap.add_argument("--no-timing", action="store_true", help="do not glFinish around passes")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument("--gl-debug", action="store_true", help="log KHR_debug messages")
    args = ap.parse_args(argv)
    if args.list_scenarios:
        return args
    if args.scenario is not None and args.scenario != "all" and args.scenario not in SCENARIOS:
        ap.error(f"unknown scenario {args.scenario!r} (have: {', '.join(sorted(SCENARIOS))}, all; "
                 "--presets FILE adds more)")
    if args.camera is not None and args.camera not in CAMERAS:
        ap.error(f"unknown camera {args.camera!r} (have: {', '.join(sorted(CAMERAS))})")
    if not args.pack and not args.shaders:
        ap.error("a shader pack is required: irisim PATH/TO/PACK [options] (or --shaders DIR)")
    if args.scenario == "all":
        return args
    sc = SCENARIOS.get(args.scenario, {})
    args.world = args.world or sc.get("world", "world0")
    args.time = args.time if args.time is not None else sc.get("time", 6000)
    args.rain = args.rain if args.rain is not None else sc.get("rain", 0.0)
    args.thunder = args.thunder if args.thunder is not None else sc.get("thunder", 0.0)
    args.camera = args.camera or sc.get("camera", "lake")
    if args.camera not in CAMERAS:
        ap.error(f"scenario {args.scenario}: unknown camera {args.camera!r}")
    args.underwater = args.underwater or sc.get("underwater", False)
    for k in STRUCTURES:
        setattr(args, k, getattr(args, k) or sc.get(k, False))
    if args.origin_dy == 0:
        args.origin_dy = sc.get("origin_dy", 0)
    args.extras = {}
    if args.glass_gallery:
        args.extras["glass_gallery"] = True
    if args.village:
        args.extras["village"] = True
    for flag, kinds, key, limit in (("--gallery", args.gallery_kinds, "gallery", 9),
                                    ("--poles", args.pole_kinds, "poles", 5)):
        if not getattr(args, key):
            continue
        lst = [k.strip() for k in kinds.split(",") if k.strip()]
        bad = [k for k in lst if k not in scene.BID or k == "air"]
        if bad or not lst or len(lst) > limit:
            ap.error(f"{flag}: 1..{limit} irisim blocks expected, got {kinds!r}"
                     + (f" (unknown: {', '.join(bad)})" if bad else ""))
        args.extras[key] = lst
    for key, wall, light in (("hall", args.hall_wall, args.hall_light), ("furnace", args.furnace_wall, args.furnace_light)):
        if not getattr(args, key):
            continue
        bad = [b for b in (wall, light) if b not in scene.BID or b == "air"]
        if bad:
            ap.error(f"--{key}: unknown irisim block {', '.join(bad)} (see BLOCKS in irisim/scene.py)")
        args.extras[key] = (wall, light)
    if args.gallery and args.poles:
        ap.error("--gallery and --poles use the same platform (x 0..41, z 0..29): pick one")
    if args.hall and args.furnace:
        ap.error("--hall and --furnace use the same area (x 60..91, z 2..33): pick one")
    args.npy_seq = {n.strip() for n in (args.npy_seq or "").split(",") if n.strip()}
    args.log_uniforms = [n.strip() for n in args.log_uniforms.split(",") if n.strip()]
    args.name = args.name or args.scenario or "custom"
    args.sets = []
    for s in args.set:
        if "=" not in s:
            ap.error(f"--set expects NAME=VALUE, got {s}")
        k, v = s.split("=", 1)
        args.sets.append((k.strip(), v.strip()))
    args.offs = args.off
    args.ons = args.on
    args.defines = {}
    for d in args.define:
        k, _, v = d.partition("=")
        args.defines[k.strip()] = v.strip()
    args.npy = [n.strip() for s in args.npy for n in s.split(",") if n.strip()]
    args.ramp = None
    if args.rain_ramp:
        try:
            a, b = (float(v) for v in args.rain_ramp.split(","))
        except ValueError:
            ap.error(f"--rain-ramp expects START,FRAMES, got {args.rain_ramp}")
        if b <= 0:
            ap.error("--rain-ramp: FRAMES must be > 0")
        args.ramp = (a, b)
    args.fog_rgb = None
    if args.fog_color:
        try:
            args.fog_rgb = np.array([float(v) for v in args.fog_color.split(",")], np.float64)
        except ValueError:
            args.fog_rgb = np.zeros(0)
        if args.fog_rgb.size != 3:
            ap.error(f"--fog-color expects R,G,B, got {args.fog_color}")
    if args.dt <= 0:
        ap.error("--dt must be > 0")
    if args.width < 1 or args.height < 1 or args.frames < 1:
        ap.error("--width, --height and --frames must be >= 1")
    args.timing = not args.no_timing
    args.dimension = DIMENSIONS[args.world]
    if args.render_distance is None:
        args.render_distance = 4 if args.dh else 8
    return args


# ============================================================================ per-frame state
class FrameState:
    def __init__(self):
        self.uniforms = {}

    def draw_matrices(self, kind, view, proj):
        """Per-draw iris_* uniforms for 'terrain' | 'entity' | 'sky' with the given base view (main or shadow).

        sodium (Iris 1.21): gl_Vertex camera-relative (region offset added in the shader), model-view = view only.
        chunk: gl_Vertex relative to the region origin, model-view includes the camera translation."""
        if kind == "sky":
            return self.matrix_uniforms(view, proj, np.zeros(3))
        # entity vertices are pre-rotated at build time (vanilla bakes the entity pose on the CPU)
        origin = self.region_origin if kind == "terrain" else self.entity_origin
        if self.vertex_space == "sodium":
            return self.matrix_uniforms(view, proj, origin - self.cam_pos)
        return self.matrix_uniforms(view @ cam.translate(origin - self.cam_pos), proj, np.zeros(3))

    def matrix_uniforms(self, mv, proj, offset=None):
        return {
            "iris_RegionOffset": np.zeros(3) if offset is None else offset,
            "iris_ModelViewMat": mv, "iris_ModelViewMatInverse": np.linalg.inv(mv),
            "iris_ProjMat": proj, "iris_ProjMatInverse": np.linalg.inv(proj),
            "iris_NormalMat": cam.normal_matrix(mv), "iris_TextureMat": np.eye(4),
            "iris_LightmapTextureMatrix": cam.LIGHTMAP_TEXTURE_MATRIX,
        }


def end_flash_view_position(view, az_el):
    """Iris endFlashPosition: the flash direction (AZ,EL degrees, azimuth clockwise from north = -Z
    towards east = +X) x 100 in view space (gbufferModelView applied, like shadowLightPosition)."""
    az, el = (math.radians(float(v)) for v in str(az_el).split(","))
    d = np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az), 0.0]) * 100.0
    return (view @ d)[:3]


def sky_and_fog(dimension, cel, rain, underwater):
    """Rough vanilla sky/fog colours."""
    if dimension == "nether":
        return np.array([0.2, 0.03, 0.03]), np.array([0.2, 0.03, 0.03])
    if dimension == "end":
        return np.array([0.0, 0.0, 0.0]), np.array([0.04, 0.03, 0.05])
    b = min(1.0, max(0.0, math.cos(cel.sky * 2 * math.pi) * 2.0 + 0.5))
    sky = np.array([0.47, 0.65, 1.0]) * b
    fog = np.array([0.75, 0.85, 1.0]) * (b * 0.94 + 0.06)
    # sunrise / sunset tint
    sd = math.cos(cel.sky * 2 * math.pi)
    if -0.4 <= sd <= 0.4:
        k = (sd / 0.4) * 0.5 + 0.5
        a = (1.0 - (1.0 - math.sin(math.pi * k)) * 0.99) ** 2
        fog = fog * (1 - a * 0.6) + np.array([1.0, 0.55, 0.3]) * a * 0.6
    if rain > 0:
        g = (sky @ np.array([0.3, 0.59, 0.11])) * 0.6
        sky = sky * (1 - rain * 0.75) + g * rain * 0.75
        fog = fog * (1 - rain * 0.5)
    if underwater:
        fog = np.array([0.05, 0.18, 0.45]) * (0.2 + 0.8 * b)
    return sky, fog


class Driver:
    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        out_root = os.path.abspath(args.out or "irisim-out")
        out_dir = os.path.join(out_root, args.name)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        # GL context first (pipeline compiles programs at construction)
        from . import glctx
        ctx = glctx.GLXContext(debug=args.gl_debug)
        from OpenGL import GL
        from . import dumps
        from . import pipeline as pl
        log = pl.Log(os.path.join(out_dir, "log.txt"), quiet=args.quiet)
        log.info(f"irisim {__version__}: {GL.glGetString(GL.GL_RENDERER).decode()} | "
                 f"{GL.glGetString(GL.GL_VERSION).decode()}")
        log.info(f"pack: {pack.SHADERS}")
        log.info(f"scenario={args.scenario} world={args.world} time={args.time} rain={args.rain} camera={args.camera} "
                 f"frames={args.frames} {args.width}x{args.height}")
        ov = []
        if args.t0:
            ov.append(f"t0={args.t0:g} s")
        if args.dt != 1.0 / 60.0:
            ov.append(f"dt={args.dt:g} s")
        if args.ramp:
            ov.append(f"rain ramp 0 -> {args.rain:g} over frames {args.ramp[0]:g}..{args.ramp[0] + args.ramp[1]:g}")
        if args.wetness is not None:
            ov.append(f"wetness={args.wetness:g}")
        if args.precipitation:
            ov.append(f"biome_precipitation={args.precipitation}")
        if args.temperature != 0.8:
            ov.append(f"temperature={args.temperature:g}")
        if args.rainfall != 0.4:
            ov.append(f"rainfall={args.rainfall:g}")
        if args.fog_rgb is not None:
            ov.append("fogColor=(" + ", ".join(f"{v:g}" for v in args.fog_rgb) + ")")
        if args.end_flash:
            ov.append(f"endFlashIntensity={args.end_flash:g} from az,el {args.end_flash_dir}"
                      + ("" if args.dimension == "end" else " (End only: 0 here)"))
        if args.test_step:
            ov.append("test step in the cave")
        if ov:
            log.info("overrides: " + "; ".join(ov))
        self.log_weather = bool(ov)
        if args.defines:
            log.info("extra macros: " + ", ".join(f"{k}={v}" if v else k for k, v in args.defines.items()))
        sx = [k if v is True else f"{k}={','.join(v)}" for k, v in args.extras.items()]
        if args.origin_dy:
            sx.append(f"origin dy {args.origin_dy:+d} (world origin y {int(scene.WORLD_ORIGIN[1])}, "
                      f"lake surface y {int(scene.WORLD_ORIGIN[1]) + scene.SEA})")
        if sx:
            log.info("test structures: " + "; ".join(sx)
                     + ("" if args.dimension == "overworld" or not args.extras else " (Overworld only: none here)"))
        if args.npy_seq:
            args.npy_seq_dir = args.npy_seq_dir or os.path.join(out_dir, "seq")
            log.info(f"npy-seq from frame {args.npy_seq_from}: {', '.join(sorted(args.npy_seq))} -> {args.npy_seq_dir}")
        if args.gl_debug:
            self._install_debug(GL, log)
        # default the shadowMapResolution option to the harness shadow size (keeps shader constants consistent)
        names = {k for k, _ in args.sets}
        decl = optmod.discover_options()
        if args.shadow_res is None:
            args.shadow_res = 0 if "shadowMapResolution" in names else 1024
        if args.shadow_res and "shadowMapResolution" in decl:
            if "shadowMapResolution" not in names:
                args.sets.append(("shadowMapResolution", str(args.shadow_res)))
                args.shadow_res = 0   # the pipeline then reads it back from the const directive
        t0 = time.perf_counter()
        self.pipe = pipe = pl.Pipeline(args, log, out_dir)
        # ---- scene
        log.section("scene")
        ts = time.perf_counter()
        # Iris block render-type overrides from block.properties (layer.translucent = minecraft:glass ...): the
        # block is meshed into that layer (gbuffers program, blending, mc_Entity.y) like Iris' BlockRenderer mixin
        for name, layer in pipe.ids["block"].layers.items():
            if name in scene.BID and layer in scene.LAYERS and name != "air":
                if scene.LAYER[scene.BID[name]] != layer:
                    log.info(f"render layer override: {name} {scene.LAYER[scene.BID[name]]} -> {layer}")
                scene.LAYER[scene.BID[name]] = layer
        world = scene.World(args.dimension, test_step=args.test_step, extras=args.extras)
        atlas = textures.Atlas(list(textures.TEXTURES), pbr=args.pbr)
        block_ids = {}
        bmap = pipe.ids["block"]
        missing = []
        for name, *_ in scene.BLOCKS[1:]:
            st = scene.BLOCK_STATE.get(name, {})
            v = bmap.lookup(name, st, default=None) if len(bmap) else None
            if v is None:
                if args.builtin_ids and not len(bmap):
                    v = scene.BUILTIN_IDS.get(name, -1)
                else:
                    v = -1
                    missing.append(name)
            block_ids[name] = v
        if not len(bmap):
            log.warn("block.properties missing or empty: mc_Entity.x = -1 for every block"
                     + (" (using --builtin-ids table)" if args.builtin_ids else " (try --builtin-ids)"))
        elif missing:
            log.info("blocks without a block.properties id (-1): " + ", ".join(missing))
        # camera (before meshing: plants right at the eye would fill the screen)
        p, tgt = CAMERAS[args.camera]
        if p[1] is None:
            p = (p[0], world._ground(int(p[0]), int(p[2])) + 1.62, p[2])
        eye = np.array([float(x) for x in args.pos.split(",")]) if args.pos else np.array(p, np.float64)
        yaw, pitch = cam.look_at_angles(p, tgt)
        if args.pos and args.yaw is None:
            yaw, pitch = cam.look_at_angles(eye, tgt)
        yaw = args.yaw if args.yaw is not None else yaw
        pitch = args.pitch if args.pitch is not None else pitch
        self.camera = cam.Camera(eye + O, yaw, pitch, args.fov)
        world.clear_plants_near(eye, 2)
        separate_ao = pipe.props.bool_flag("separateAo", False)
        meshes = scene.build_world_mesh(world, atlas, block_ids, separate_ao=separate_ao)
        self.sky_verts = scene.build_sky_mesh()
        meshes["sky"] = self.sky_verts
        # entity placement (feet on the ground near the lake shore)
        ex, ez = 46, 67
        ey = world._ground(ex, ez)
        self.entity_pos = np.array([ex + 0.5, ey, ez + 0.5]) + O
        self.entity_yaw = 140.0
        if not args.no_entity:
            ev = scene.build_entity_mesh(atlas, light=world.light_at((ex + 0.5, ey + 0.5, ez + 0.5)))
            R = cam.rot_y(-self.entity_yaw)[:3, :3]
            for a, b in ((0, 3), (11, 14), (18, 21), (22, 25)):
                ev[:, a:b] = ev[:, a:b] @ R.T
            meshes["entity"] = ev
        if args.dh:
            meshes.update(scene.build_dh_meshes(args.dimension, radius=args.dh_distance))
        lm = textures.vanilla_lightmap(1.0, args.dimension)
        log.info(f"world {scene.SX}x{scene.SY}x{scene.SZ} ({args.dimension}) quads: "
                 + ", ".join(f"{k}={len(v) // 4}" for k, v in meshes.items()) + f"  separateAo={separate_ao}  "
                 f"built in {time.perf_counter() - ts:.1f} s")
        pipe.create_resources(atlas, lm, meshes)
        self.world, self.atlas = world, atlas
        entity_ids = pipe.ids["entity"]
        self.entity_id = entity_ids.lookup("pig", default=-1) if len(entity_ids) else -1
        self.held = (-1, 0)
        if args.held:
            iid = pipe.ids["item"].lookup(args.held, default=-1) if len(pipe.ids["item"]) else -1
            lv = {"torch": 14, "lantern": 15, "glowstone": 15, "soul_torch": 10, "lava_bucket": 15}.get(args.held, 0)
            self.held = (iid, lv)
        log.info(f"setup done in {time.perf_counter() - t0:.1f} s")

        # ---- frames
        log.section("frames")
        prev = None
        self.center_depth = None
        self.frame_ms = []
        for f in range(args.frames):
            last = f == args.frames - 1
            dump = args.dump == "all" or (args.dump == "last" and last)
            if dump:
                ddir = out_dir if last else os.path.join(out_dir, f"frame{f:03d}")
                pipe.dumper = dumps.Dumper(os.path.join(ddir, "passes") if last else ddir, log)
                log.info(f"-- frame {f} (dumping to {os.path.relpath(pipe.dumper.dir, out_root)})")
            fs = self.frame_state(f, prev)
            pipe.frame_index = f
            pipe.timings_frame_start = {k: len(v) for k, v in pipe.timings.items()}
            tf = time.perf_counter()
            d0 = pipe.dumper.seconds if (dump and pipe.dumper) else 0.0
            pipe.render_frame(fs, dump=dump)
            dumped = (pipe.dumper.seconds - d0) if (dump and pipe.dumper) else 0.0
            dt = (time.perf_counter() - tf - dumped) * 1000
            self.frame_ms.append(dt)
            cd = pipe.center_depth
            hl = pipe.half_lives["centerDepthHalflife"]
            if self.center_depth is None:
                self.center_depth = cd
            else:
                a = 1.0 - math.exp(-math.log(2) * fs.uniforms["frameTime"] / max(hl * 0.1, 1e-3))
                self.center_depth += (cd - self.center_depth) * a
            img = pipe.read_output()
            if args.npy_seq and f >= args.npy_seq_from:
                os.makedirs(args.npy_seq_dir, exist_ok=True)
                np.save(os.path.join(args.npy_seq_dir, f"final_f{f:03d}.npy"), np.ascontiguousarray(img[..., :3]))
            from PIL import Image
            if last:
                Image.fromarray(np.ascontiguousarray(img[..., :3])).save(os.path.join(out_dir, "final.png"))
            if args.dump == "all" or args.save_frames:
                Image.fromarray(np.ascontiguousarray(img[..., :3])).save(os.path.join(out_dir, f"frame{f:03d}.png"))
            U = fs.uniforms
            weather = ""
            if self.log_weather:
                # the uniforms the overrides feed
                weather = (f"  | t {U['frameTimeCounter']:.3f} rain {U['rainStrength']:.3f} wetness {U['wetness']:.3f}"
                           f" precip {U['biome_precipitation']}"
                           + "".join(f" {k} {float(U[k]):.4f}" for k in LOG_UNIFORMS if k in U))
            if args.log_uniforms:
                weather += "  |" + "".join(f" {k} {_fmt_uniform(U.get(k))}" for k in args.log_uniforms)
            log.info(f"frame {f}: {dt:7.0f} ms" + (f" (+{dumped:.1f} s dumping)" if dumped else "")
                     + f"  center depth {cd:.5f}  final mean rgb "
                     f"{tuple(int(x) for x in img[..., :3].reshape(-1, 3).mean(0))}" + weather)
            if last and args.npy:
                pipe.dump_npy(args.npy, out_dir)
            prev = fs
            # advance camera
            if args.motion or args.turn:
                self.camera.pos = self.camera.pos + self.camera.forward() * args.motion
                self.camera.yaw += args.turn
        self.report(log, out_dir)
        ctx.release()
        return 0 if not pipe.failed else 2

    # ------------------------------------------------------------------
    def frame_state(self, f, prev):
        args, pipe, world = self.args, self.pipe, self.world
        fs = FrameState()
        c = self.camera
        dim = args.dimension
        wt = int(args.time + args.time_speed * f)
        cel = cam.Celestial(wt, pipe.sun_path_rotation)
        view = c.view()
        aspect = args.width / args.height
        far_blocks = args.render_distance * 16
        proj = cam.perspective(c.fov, aspect, 0.05, far_blocks * 4.0)
        pos = c.pos
        pos_int = np.floor(pos).astype(np.int64)
        frac = pos - pos_int
        sun, moon, shadow_light, up = cel.positions(view)
        smv = cel.shadow_model_view(pos, pipe.shadow_interval)
        sproj = cam.ortho_symmetric(pipe.shadow_distance * 2, pipe.shadow_distance * 2, pipe.shadow_near,
                                    pipe.shadow_far)
        eye_idx = pos - O
        bl, sk = world.light_at(eye_idx)
        blk_name = world.block_at(eye_idx)
        in_water = 1 if (args.underwater or blk_name == "water") else (2 if blk_name == "lava" else 0)
        frame_time = args.dt
        t_now = args.t0 + f * frame_time
        rain_now = args.rain
        if args.ramp:
            rain_now = args.rain * min(max((f - args.ramp[0]) / args.ramp[1], 0.0), 1.0)
        wet_now = rain_now if args.wetness is None else args.wetness
        sky_col, fog_col = sky_and_fog(dim, cel, rain_now, in_water == 1)
        if args.fog_rgb is not None:
            fog_col = args.fog_rgb.copy()
        if args.precipitation is None:
            precip = 1 if dim == "overworld" else 0
        else:
            precip = {"none": 0, "rain": 1, "snow": 2}[args.precipitation]
        U = {
            "frameCounter": f, "frameTime": frame_time, "frameTimeCounter": t_now % 3600.0,
            "worldTime": wt % 24000, "worldDay": wt // 24000, "moonPhase": (wt // 24000) % 8,
            "sunAngle": cel.sun_angle, "shadowAngle": cel.shadow_angle,
            "viewWidth": float(args.width), "viewHeight": float(args.height), "aspectRatio": aspect,
            "near": 0.05, "far": float(far_blocks),
            "centerDepthSmooth": self.center_depth if self.center_depth is not None else 1.0,
            "cameraPosition": pos, "cameraPositionInt": pos_int, "cameraPositionFract": frac,
            "eyePosition": pos, "eyeAltitude": pos[1], "firstPersonCamera": True,
            "gbufferModelView": view, "gbufferModelViewInverse": np.linalg.inv(view),
            "gbufferProjection": proj, "gbufferProjectionInverse": np.linalg.inv(proj),
            "shadowModelView": smv, "shadowModelViewInverse": np.linalg.inv(smv),
            "shadowProjection": sproj, "shadowProjectionInverse": np.linalg.inv(sproj),
            "sunPosition": sun, "moonPosition": moon, "shadowLightPosition": shadow_light, "upPosition": up,
            "rainStrength": rain_now, "wetness": wet_now, "thunderStrength": args.thunder,
            "lightningBoltPosition": np.zeros(4), "biome_precipitation": precip,
            "temperature": args.temperature, "rainfall": args.rainfall, "fogColor": fog_col, "skyColor": sky_col,
            # Iris (MC 1.21.9+): End sky flash; endFlashPosition is view space, length 100, End only
            "endFlashIntensity": float(args.end_flash) if dim == "end" else 0.0,
            "endFlashPosition": end_flash_view_position(view, args.end_flash_dir) if dim == "end" else np.zeros(3),
            "hasSkylight": dim == "overworld", "hasCeiling": dim == "nether",
            "ambientLight": 0.1 if dim == "nether" else 0.0,
            "bedrockLevel": -64 if dim == "overworld" else 0, "heightLimit": 384 if dim == "overworld" else 256,
            "cloudHeight": 192.33,
            "eyeBrightness": np.array([bl * 16, sk * 16]), "eyeBrightnessSmooth": np.array([bl * 16, sk * 16]),
            "isEyeInWater": in_water, "blindness": 0.0, "darknessFactor": 0.0, "darknessLightFactor": 0.0,
            "nightVision": 0.0, "screenBrightness": 0.5, "heldItemId": self.held[0], "heldItemId2": -1,
            "heldBlockLightValue": self.held[1], "heldBlockLightValue2": 0, "isRightHanded": True,
            "is_sneaking": False, "is_sprinting": False, "is_hurt": False, "is_invisible": False,
            "is_burning": False, "is_on_ground": True, "playerMood": 0.0, "constantMood": 0.0,
            "renderStage": 0, "entityId": -1, "blockEntityId": -1, "currentRenderedItemId": -1,
            "entityColor": np.zeros(4), "atlasSize": np.array(pipe.atlas_size), "currentColorSpace": 0,
            "hideGUI": True, "pi": math.pi, "dimension": {"overworld": 0, "nether": -1, "end": 1}[dim],
        }
        if prev is None:
            # Iris: previous values start at zero / identity on the first frame
            U.update({"previousCameraPosition": np.zeros(3), "previousCameraPositionInt": np.zeros(3, np.int64),
                      "previousCameraPositionFract": np.zeros(3), "gbufferPreviousModelView": np.eye(4),
                      "gbufferPreviousProjection": np.eye(4)})
        else:
            pu = prev.uniforms
            U.update({"previousCameraPosition": pu["cameraPosition"], "previousCameraPositionInt": pu["cameraPositionInt"],
                      "previousCameraPositionFract": pu["cameraPositionFract"],
                      "gbufferPreviousModelView": pu["gbufferModelView"],
                      "gbufferPreviousProjection": pu["gbufferProjection"]})
        # Distant Horizons (Iris CommonUniforms / MatrixUniforms / DHCompatInternal): far plane
        # (dhRenderDistance + 512) * sqrt(2), the DH projection = gbufferProjection's FOV / aspect with the
        # DH near / far planes; previous values start at identity / zero like the other "previous" ones.
        # (without DH, Iris' DHCompat returns near = far = 0.01 and dhProjection = gbufferProjection)
        dh_near, dh_far = 0.5, (args.dh_distance + 512) * math.sqrt(2.0)
        dh_proj = cam.perspective(c.fov, aspect, dh_near, dh_far) if args.dh else proj
        U.update({"dhRenderDistance": args.dh_distance if args.dh else args.render_distance,
                  "dhNearPlane": dh_near if args.dh else 0.01, "dhFarPlane": dh_far if args.dh else 0.01,
                  "dhProjection": dh_proj, "dhProjectionInverse": np.linalg.inv(dh_proj),
                  "dhPreviousProjection": np.eye(4) if prev is None else prev.uniforms["dhProjection"]})
        U.update(pipe.custom_uniforms(U) if hasattr(pipe, "custom_uniforms") else {})
        fs.uniforms = U
        fs.dh_proj = dh_proj
        fs.view, fs.proj = view, proj
        fs.vertex_space = args.vertex_space
        fs.cam_pos = pos
        fs.region_origin = O
        fs.cam_region = pos - O
        fs.entity_origin = self.entity_pos
        fs.entity_id = self.entity_id
        fs.shadow_mv, fs.shadow_proj = smv, sproj
        fs.draw_sky = dim == "overworld" or dim == "end"
        fs.sky_program = "gbuffers_skytextured" if dim == "end" else "gbuffers_skybasic"
        # sky colour reaches gl_Color through the colour modulator
        sv = self.sky_verts.copy()
        if dim == "end":
            # vanilla end sky: end_sky texture tinted (40, 40, 40)/255
            tx, ty = self.atlas.index["obsidian"]
            sv[:, 7] = (tx + 0.5) / self.atlas.grid
            sv[:, 8] = (ty + 0.5) / self.atlas.grid
            sv[:, 3:6] = 40.0 / 255.0
        else:
            sv[:, 3:6] = sky_col
        pipe.meshes["sky"].update_vertices(sv)
        fs.rain_mesh = None
        if rain_now > 0 and dim == "overworld" and precip != 0:
            rv = scene.build_rain_mesh(world, pos - O, self.atlas, t=t_now)
            if not hasattr(self, "_rain_mesh"):
                from . import pipeline as pl
                self._rain_mesh = pl.Mesh(rv)
            fs.rain_mesh = self._rain_mesh
        return fs

    def report(self, log, out_dir):
        pipe = self.pipe
        n = self.args.frames
        show = min(n, 8)
        log.section("timings in ms (frame 0 includes llvmpipe shader JIT; per-frame columns for the last "
                    f"{show} frames, then mean of frames 1..)")
        hdr = "".join(f"{'f' + str(n - show + i):>8s}" for i in range(show))
        log.info(f"  {'pass':24s}{hdr} {'mean1+':>9s}")
        for k, v in pipe.timings.items():
            cols = "".join(f"{x:8.1f}" for x in v[-show:]) if len(v) == n else (f"{v[0]:8.1f}" + " " * 8 * (show - 1))
            rest = v[1:] if len(v) > 1 else v
            log.info(f"  {k:24s}{cols:{8 * show}s} {sum(rest) / len(rest):9.1f}")
        rest = self.frame_ms[1:] or self.frame_ms
        log.info(f"  {'frame total':24s}" + "".join(f"{x:8.0f}" for x in self.frame_ms[-show:])
                 + f" {sum(rest) / len(rest):9.1f}")
        if pipe.missing_uniforms:
            log.section("uniforms declared by programs but not provided by irisim")
            for k, v in sorted(pipe.missing_uniforms.items()):
                log.info(f"  {k:28s} used by {', '.join(sorted(v)[:6])}{' ...' if len(v) > 6 else ''}")
        if pipe.missing_samplers:
            log.section("samplers/images without a resource (bound to a black dummy)")
            for k, v in sorted(pipe.missing_samplers.items()):
                log.info(f"  {k:28s} used by {', '.join(sorted(v)[:6])}{' ...' if len(v) > 6 else ''}")
        log.section("summary")
        log.info(f"  failed programs: {len(pipe.failed)}" + (" (" + ", ".join(pipe.failed) + ")" if pipe.failed else ""))
        log.info(f"  warnings: {log.counts['warn']}  errors: {log.counts['error']}")
        log.info(f"  output: {os.path.join(out_dir, 'final.png')}")

    @staticmethod
    def _install_debug(GL, log):
        try:
            from OpenGL.GL import GLDEBUGPROC

            def cb(source, typ, ident, severity, length, message, user):
                if severity == GL.GL_DEBUG_SEVERITY_NOTIFICATION:
                    return
                m = message[:length].decode(errors="replace") if isinstance(message, bytes) else str(message)
                log.once(("gldbg", m), "GLDEBUG " + m)
            Driver._dbg_cb = GLDEBUGPROC(cb)
            GL.glEnable(GL.GL_DEBUG_OUTPUT)
            GL.glEnable(GL.GL_DEBUG_OUTPUT_SYNCHRONOUS)
            GL.glDebugMessageCallback(Driver._dbg_cb, None)
        except Exception as e:  # noqa: BLE001
            log.warn(f"GL debug output unavailable: {e}")


def shift_world_origin(dy):
    """--origin-dy: move the scene dy blocks in world y (camera, entity, rain and DH LODs follow, they all
    use the origin; the block-index coordinates of --pos and the presets stay the same)."""
    global O
    scene.WORLD_ORIGIN = scene.WORLD_ORIGIN + np.array([0, int(dy), 0], np.int64)
    O = scene.WORLD_ORIGIN.astype(np.float64)


def _fmt_uniform(v):
    if v is None:
        return "n/a"
    a = np.asarray(v, dtype=np.float64).ravel()
    return f"{a[0]:.4f}" if a.size == 1 else "(" + ", ".join(f"{x:.4f}" for x in a[:4]) + (", ..)" if a.size > 4 else ")")


def _list(args, shaders):
    if args.list_scenarios:
        print("scenarios:")
        for k, v in SCENARIOS.items():
            print(f"  {k:12s} " + ", ".join(f"{a}={b}" for a, b in v.items()))
        print("  all          " + " ".join(ALL_SCENARIOS))
        print("cameras:")
        for k, (e, t) in CAMERAS.items():
            print(f"  {k:12s} eye {e}  target {t}")
        return 0
    decl = optmod.discover_options(shaders)
    for name in sorted(decl, key=lambda n: (decl[n].file, n)):
        o = decl[name]
        vals = "" if o.kind == "bool" else "  [" + " ".join(o.values) + "]"
        print(f"{o.file:40s} {o.kind:6s} {name} = {o.default}{vals}")
    print(f"{len(decl)} options", file=sys.stderr)
    return 0


def _maybe_reexec_under_xvfb(args, argv):
    """Re-run this command under xvfb-run when there is no X display (GLX needs one)."""
    if os.environ.get("IRISIM_IN_XVFB") == "1" or args.xvfb == "never":
        return None
    if args.xvfb == "auto" and os.environ.get("DISPLAY"):
        return None
    if shutil.which("xvfb-run") is None:
        raise SystemExit("irisim: no DISPLAY and xvfb-run is not installed (apt install xvfb), or run with "
                         "DISPLAY set and --xvfb never")
    env = dict(os.environ, IRISIM_IN_XVFB="1")
    cmd = ["xvfb-run", "-a", "-s", "-screen 0 640x480x24", sys.executable, "-m", "irisim"] + list(argv)
    return subprocess.call(cmd, env=env)


def _configure_host(args):
    if args.mc_version:
        optmod.HOST["mc_version"] = args.mc_version
    if args.gl_vendor:
        optmod.HOST["vendor"] = args.gl_vendor
        optmod.HOST["renderer"] = {"NVIDIA": "GEFORCE", "AMD": "RADEON", "ATI": "RADEON", "INTEL": "INTEL",
                                   "MESA": "MESA"}.get(args.gl_vendor, "OTHER")
    if args.os:
        optmod.HOST["os"] = args.os
    pack.EXTRA_MACROS.clear()
    pack.EXTRA_MACROS.update(args.defines)


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    args = parse_args(argv)
    if args.list_scenarios:
        return _list(args, None)
    tmp = []
    try:
        if args.list_options:
            return _list(args, os.path.abspath(args.shaders) if args.shaders else find_shaders_dir(args.pack, tmp))
        os.environ.setdefault("LP_NUM_THREADS", str(max(1, min(8, (os.cpu_count() or 2) - 1))))
        r = _maybe_reexec_under_xvfb(args, argv)
        if r is not None:
            return r
        shaders = os.path.abspath(args.shaders) if args.shaders else find_shaders_dir(args.pack, tmp)
        if args.scenario == "all":
            # one process per scenario (module state such as the world origin is per run)
            rc = 0
            base = []
            skip = False
            for a in argv:
                if skip:
                    skip = False
                    continue
                if a in ("--scenario", "--shaders"):
                    skip = True
                    continue
                if a.startswith(("--scenario=", "--shaders=")):
                    continue
                if a == args.pack:
                    continue
                base.append(a)
            for sname in ALL_SCENARIOS:
                print(f"=== scenario {sname}", flush=True)
                r = subprocess.call([sys.executable, "-m", "irisim", "--shaders", shaders, "--scenario", sname]
                                    + base, env=dict(os.environ))
                rc = rc or r
            return rc
        if args.origin_dy:
            shift_world_origin(args.origin_dy)
        pack.set_shaders_dir(shaders)
        _configure_host(args)
        return Driver(args).run()
    finally:
        for t in tmp:
            shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
