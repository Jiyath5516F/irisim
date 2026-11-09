"""Shader pack options and Iris standard macros, without Minecraft.

Option discovery follows Iris' OptionAnnotatedSource (net.irisshaders.iris.shaderpack.option):

* every shader source file of the pack is scanned line by line (irisim scans every file with a
  shader extension under the ``shaders/`` directory; Iris scans the files reachable from its
  programs, which is the same set for any pack without dead files);
* a **boolean option** is ``#define NAME`` or ``//#define NAME`` (optionally followed by a ``//``
  comment), and only counts when ``NAME`` is referenced by an ``#ifdef NAME`` / ``#ifndef NAME`` line
  somewhere in the pack (Iris ignores ``#if defined(NAME)`` for this purpose too);
* a **value option** is ``#define NAME VALUE // [v1 v2 ...]`` (no leading ``//``): the comment must
  contain an allowed-values list in brackets;
* a **const option** is ``const TYPE NAME = VALUE; // [v1 v2 ...]`` (or ``const bool NAME = true;``)
  where ``NAME`` is one of Iris' configurable const directives (``shadowMapResolution``,
  ``sunPathRotation``, ...: :data:`VALID_CONST_OPTION_NAMES`).

Overrides (``--set``, ``--on``, ``--off``) are applied the way Iris applies a changed option value: the
declaring line is rewritten (booleans commented / uncommented, values replaced) in every file that
declares the option.
"""
import os
import re

SHADER_EXTENSIONS = (".glsl", ".vsh", ".fsh", ".gsh", ".csh", ".tcs", ".tes", ".vert", ".frag", ".geom", ".comp",
                     ".inc")

RENDER_STAGES = ["NONE", "SKY", "SUNSET", "CUSTOM_SKY", "SUN", "MOON", "STARS", "VOID", "TERRAIN_SOLID",
                 "TERRAIN_CUTOUT_MIPPED", "TERRAIN_CUTOUT", "ENTITIES", "BLOCK_ENTITIES", "DESTROY", "OUTLINE",
                 "DEBUG", "HAND_SOLID", "TERRAIN_TRANSLUCENT", "TRIPWIRE", "PARTICLES", "CLOUDS", "RAIN_SNOW",
                 "WORLD_BORDER", "HAND_TRANSLUCENT"]
FEATURES = ["SEPARATE_HARDWARE_SAMPLERS", "PER_BUFFER_BLENDING", "COMPUTE_SHADERS", "ENTITY_TRANSLUCENT", "SSBO",
            "CUSTOM_IMAGES", "HIGHER_SHADOWCOLOR", "REVERSED_CULLING", "BLOCK_EMISSION_ATTRIBUTE"]
# Iris StandardMacros defines the DH block materials always, DISTANT_HORIZONS only while DH renders.
DH_BLOCKS = ["UNKNOWN", "LEAVES", "STONE", "WOOD", "METAL", "DIRT", "LAVA", "DEEPSLATE", "SNOW", "SAND", "TERRACOTTA",
             "NETHER_STONE", "WATER", "GRASS", "AIR", "ILLUMINATED"]

# The "host" irisim pretends to be. Change with --mc-version / --iris-version / --gl-vendor / --os.
HOST = {
    "mc_version": 12101,       # MC_VERSION (1.21.1)
    "iris_version": 10800,     # IRIS_VERSION (1.8.0)
    "vendor": "NVIDIA",        # MC_GL_VENDOR_<X>: NVIDIA, AMD, INTEL, MESA, ATI, OTHER
    "renderer": "GEFORCE",     # MC_GL_RENDERER_<X>: GEFORCE, RADEON, INTEL, MESA, ...
    "os": "LINUX",             # MC_OS_<X>: LINUX, WINDOWS, MAC, UNKNOWN
}

VALID_CONST_OPTION_NAMES = {
    "shadowMapResolution", "shadowDistance", "voxelDistance", "shadowDistanceRenderMul", "entityShadowDistanceMul",
    "shadowIntervalSize", "generateShadowMipmap", "generateShadowColorMipmap", "shadowHardwareFiltering",
    "shadowtex0Mipmap", "shadowtexMipmap", "shadowtex1Mipmap", "shadowtex0Nearest", "shadowtexNearest",
    "shadow0MinMagNearest", "shadowtex1Nearest", "shadow1MinMagNearest", "wetnessHalflife", "drynessHalflife",
    "eyeBrightnessHalflife", "centerDepthHalflife", "sunPathRotation", "ambientOcclusionLevel", "superSamplingLevel",
    "noiseTextureResolution",
}
for _i in range(8):
    VALID_CONST_OPTION_NAMES |= {f"shadowcolor{_i}Mipmap", f"shadowColor{_i}Mipmap", f"shadowcolor{_i}Nearest",
                                 f"shadowColor{_i}Nearest", f"shadowcolor{_i}MinMagNearest",
                                 f"shadowColor{_i}MinMagNearest", f"shadowHardwareFiltering{_i}"}

SHADERS = None   # set by pack.set_shaders_dir()


def standard_macros(dh=False, extra=None):
    """Iris StandardMacros as '#define ...' lines (plus user --define macros)."""
    h = HOST
    m = [
        f"MC_VERSION {h['mc_version']}", "MC_MIPMAP_LEVEL 4", f"IRIS_VERSION {h['iris_version']}",
        "MC_GL_VERSION 450", "MC_GLSL_VERSION 450", f"MC_OS_{h['os']}", f"MC_GL_VENDOR_{h['vendor']}",
        f"MC_GL_RENDERER_{h['renderer']}", "IS_IRIS", "IRIS_REQUIRES_SEPARATE_ENTITY_DRAWS", "MAX_COLOR_BUFFERS 16",
        "IRIS_HAS_TRANSLUCENCY_SORTING", "IRIS_TAG_SUPPORT 2", "MC_NORMAL_MAP", "MC_SPECULAR_MAP",
        "MC_RENDER_QUALITY 1.0", "MC_SHADOW_QUALITY 1.0", "MC_HAND_DEPTH 0.125",
    ]
    for i, s in enumerate(RENDER_STAGES):
        m.append(f"MC_RENDER_STAGE_{s} {i}")
    for f in FEATURES:
        m.append(f"IRIS_FEATURE_{f}")
    if dh:
        m.append("DISTANT_HORIZONS")
    for i, b in enumerate(DH_BLOCKS):
        m.append(f"DH_BLOCK_{b} {i}")
    for name, val in (extra or {}).items():
        m.append(f"{name} {val}".rstrip())
    return ["#define " + x for x in m]


# ---------------------------------------------------------------- discovery
RE_BOOL = re.compile(r"^(\s*)(//)?\s*#define\s+([A-Za-z_]\w*)\s*(//.*)?$")
RE_VALUE = re.compile(r"^(\s*)#define\s+([A-Za-z_]\w*)\s+([\w.+-]+)\s*//(.*?)\[([^\]]*)\](.*)$")
RE_CONST = re.compile(r"^(\s*)const\s+(int|float|bool)\s+([A-Za-z_]\w*)\s*=\s*([^;]+);\s*(//.*)?$")
RE_IFDEF = re.compile(r"^\s*#\s*if(?:n)?def\s+([A-Za-z_]\w*)\s*$")


class Option:
    """kind: 'bool' (define or const bool), 'value' (#define with values), 'const' (const int/float)."""

    def __init__(self, name, kind, default, values, file):
        self.name, self.kind, self.default, self.values, self.file = name, kind, default, values, file

    def __repr__(self):
        return f"Option({self.name!r}, {self.kind}, default={self.default!r}, {len(self.values)} values, {self.file})"


def source_files(shaders_dir=None):
    root = shaders_dir or SHADERS
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.endswith(SHADER_EXTENSIONS):
                yield os.path.join(dirpath, fn)


def _read_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().replace("\r\n", "\n").split("\n")


def discover_options(shaders_dir=None):
    """name -> Option, for the whole pack (first declaration wins, like Iris for consistent duplicates)."""
    root = shaders_dir or SHADERS
    cands = []          # (name, kind, default, values, relfile)
    referenced = set()  # names on #ifdef / #ifndef lines
    for path in source_files(root):
        rel = os.path.relpath(path, root)
        for line in _read_lines(path):
            if "#define" not in line and "const" not in line and "#if" not in line:
                continue
            mi = RE_IFDEF.match(line)
            if mi:
                referenced.add(mi.group(1))
                continue
            mv = RE_VALUE.match(line)
            if mv:
                cands.append((mv.group(2), "value", mv.group(3), mv.group(5).split(), rel))
                continue
            mc = RE_CONST.match(line)
            if mc and mc.group(3) in VALID_CONST_OPTION_NAMES:
                typ, val, comment = mc.group(2), mc.group(4).strip(), mc.group(5) or ""
                if typ == "bool" and val in ("true", "false"):
                    cands.append((mc.group(3), "bool", val == "true", [True, False], rel))
                else:
                    mvals = re.search(r"\[([^\]]*)\]", comment)
                    if mvals:
                        cands.append((mc.group(3), "const", val, mvals.group(1).split(), rel))
                continue
            mb = RE_BOOL.match(line)
            if mb:
                cands.append((mb.group(3), "define_bool", mb.group(2) is None, [True, False], rel))
    opts = {}
    for name, kind, default, values, rel in cands:
        if kind == "define_bool":
            if name not in referenced:
                continue
            kind = "bool"
        if name not in opts:
            opts[name] = Option(name, kind, default, values, rel)
    return opts


def apply_option_overrides(line, overrides):
    """Rewrite one source line if it declares an overridden option (Iris OptionAnnotatedSource.edit)."""
    if not overrides or ("#define" not in line and "const" not in line):
        return line
    mv = RE_VALUE.match(line)
    if mv and mv.group(2) in overrides:
        return (f"{mv.group(1)}#define {mv.group(2)} {overrides[mv.group(2)]} //{mv.group(4)}[{mv.group(5)}]"
                f"{mv.group(6)}")
    mc = RE_CONST.match(line)
    if mc and mc.group(3) in overrides:
        v = overrides[mc.group(3)]
        if isinstance(v, bool):
            v = "true" if v else "false"
        tail = (" " + mc.group(5)) if mc.group(5) else ""
        return f"{mc.group(1)}const {mc.group(2)} {mc.group(3)} = {v};{tail}"
    mb = RE_BOOL.match(line)
    if mb and mb.group(3) in overrides:
        on = overrides[mb.group(3)]
        if isinstance(on, str):
            on = on.strip().lower() not in ("0", "false", "off", "no")
        rest = (" " + mb.group(4)) if mb.group(4) else ""
        return f"{mb.group(1)}{'' if on else '//'}#define {mb.group(3)}{rest}"
    return line
