"""Shader pack loading the way Iris does it, without Minecraft.

* options: discovered from the pack's sources like Iris (options.py), overridden from the CLI
* sources: include resolution (with a line map for error messages), option
  rewriting, Iris standard macros, #extension hoisting, compatibility-builtin
  patching (gbuffers/shadow vs composite style)
* directives: const / comment directives read from the *active* preprocessed
  text (inactive #if branches removed, comments kept — like Iris+JCPP)
* properties: shaders.properties / block / item / entity .properties, run
  through the same preprocessor with the option values as macros
* program discovery: generic, from the files present in shaders/world*/
"""
import os
import re

from . import cpp
from . import options

SHADERS = None
RENDER_STAGES = list(options.RENDER_STAGES)
EXTRA_MACROS = {}   # --define NAME[=VALUE]


def set_shaders_dir(path):
    """Point irisim at a pack's shaders/ directory."""
    global SHADERS
    SHADERS = os.path.abspath(path)
    options.SHADERS = SHADERS


STAGE_INDEX = {s: i for i, s in enumerate(RENDER_STAGES)}

COMPOSITE_GROUPS = ["shadowcomp", "begin", "prepare", "deferred", "composite"]
SHADOW_PROGRAMS = ["shadow", "shadow_solid", "shadow_cutout", "shadow_water", "shadow_entities", "shadow_block",
                   "shadow_lightning"]

# Iris ProgramId fallbacks (gbuffers group) — name -> fallback name
GBUFFER_FALLBACK = {
    "gbuffers_basic": None,
    "gbuffers_line": "gbuffers_basic",
    "gbuffers_textured": "gbuffers_basic",
    "gbuffers_textured_lit": "gbuffers_textured",
    "gbuffers_skybasic": "gbuffers_basic",
    "gbuffers_skytextured": "gbuffers_textured",
    "gbuffers_clouds": "gbuffers_textured",
    "gbuffers_terrain": "gbuffers_textured_lit",
    "gbuffers_terrain_solid": "gbuffers_terrain",
    "gbuffers_terrain_cutout": "gbuffers_terrain",
    "gbuffers_damagedblock": "gbuffers_terrain",
    "gbuffers_block": "gbuffers_terrain",
    "gbuffers_block_translucent": "gbuffers_block",
    "gbuffers_beaconbeam": "gbuffers_textured",
    "gbuffers_item": "gbuffers_textured_lit",
    "gbuffers_entities": "gbuffers_textured_lit",
    "gbuffers_entities_translucent": "gbuffers_entities",
    "gbuffers_lightning": "gbuffers_entities",
    "gbuffers_particles": "gbuffers_textured_lit",
    "gbuffers_particles_translucent": "gbuffers_particles",
    "gbuffers_entities_glowing": "gbuffers_entities",
    "gbuffers_armor_glint": "gbuffers_textured",
    "gbuffers_spidereyes": "gbuffers_textured",
    "gbuffers_hand": "gbuffers_textured_lit",
    "gbuffers_weather": "gbuffers_textured_lit",
    "gbuffers_water": "gbuffers_terrain",
    "gbuffers_hand_water": "gbuffers_hand",
    "shadow": None,
    "shadow_solid": "shadow",
    "shadow_cutout": "shadow",
    "shadow_water": "shadow",
    "shadow_entities": "shadow",
    "shadow_lightning": "shadow_entities",
    "shadow_block": "shadow",
    # Distant Horizons (Iris ProgramId: DhWater / DhGeneric fall back to DhTerrain; DhShadow has none)
    "dh_terrain": None,
    "dh_water": "dh_terrain",
    "dh_generic": "dh_terrain",
}
DH_PROGRAMS = ["dh_terrain", "dh_water"]



# ============================================================================ options
class Options:
    def __init__(self, sets, offs, ons):
        self.decl = options.discover_options()
        self.overrides = {}
        self.unknown = []
        for name, val in sets:
            o = self.decl.get(name)
            if o is None:
                self.unknown.append(name)
            if o is not None and o.kind == "bool":
                self.overrides[name] = val.lower() not in ("0", "false", "off", "no")
            else:
                self.overrides[name] = val
        for name in offs:
            if name not in self.decl:
                self.unknown.append(name)
            self.overrides[name] = False
        for name in ons:
            if name not in self.decl:
                self.unknown.append(name)
            self.overrides[name] = True

    def value(self, name):
        o = self.decl.get(name)
        if name in self.overrides:
            return self.overrides[name]
        return None if o is None else o.default

    def macros(self):
        """Option values as preprocessor macros (Iris: enabled booleans defined, values as text)."""
        m = {}
        for name, o in self.decl.items():
            v = self.value(name)
            if o.kind == "bool":
                if v:
                    m[name] = "1"
            else:
                m[name] = str(v)
        return m


def standard_macro_dict(dh=False):
    d = {}
    for line in options.standard_macros(dh, EXTRA_MACROS):
        parts = line.split(None, 2)
        d[parts[1]] = parts[2] if len(parts) > 2 else ""
    return d


# ============================================================================ sources
RE_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$')


class SourceError(Exception):
    pass


class Source:
    """A fully include-resolved program stage."""

    def __init__(self, path, lines, origins):
        self.path = path
        self.lines = lines          # list[str]
        self.origins = origins      # list[(relfile, lineno)]
        self.compile_text = None
        self.active = None          # active text (comments kept) for directive parsing

    def origin(self, line_no):
        """Map a 1-based line of compile_text back to file:line."""
        if self.compile_origins and 1 <= line_no <= len(self.compile_origins):
            f, n = self.compile_origins[line_no - 1]
            return f"{f}:{n}"
        return f"?:{line_no}"


def _resolve(path, overrides, stack, lines, origins, depth=0):
    if depth > 32:
        raise SourceError("include depth > 32 at " + path)
    if not os.path.isfile(path):
        raise SourceError(f"missing file {os.path.relpath(path, SHADERS)} (included from {stack[-1] if stack else '-'})")
    rel = os.path.relpath(path, SHADERS)
    with open(path, encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")
    for n, line in enumerate(text.split("\n"), 1):
        m = RE_INCLUDE.match(line)
        if m:
            inc = m.group(1)
            target = os.path.join(SHADERS, inc.lstrip("/")) if inc.startswith("/") else os.path.join(os.path.dirname(path), inc)
            _resolve(os.path.normpath(target), overrides, stack + [rel], lines, origins, depth + 1)
            continue
        line = options.apply_option_overrides(line, overrides)
        lines.append(line)
        origins.append((rel, n))


def load_stage(path, overrides):
    lines, origins = [], []
    _resolve(path, overrides, [], lines, origins)
    return Source(path, lines, origins)


# ---------------------------------------------------------------- compatibility patching
_MAT_DECLS = {
    "iris_ModelViewMat": "uniform mat4 iris_ModelViewMat;",
    "iris_ModelViewMatInverse": "uniform mat4 iris_ModelViewMatInverse;",
    "iris_ProjMat": "uniform mat4 iris_ProjMat;",
    "iris_ProjMatInverse": "uniform mat4 iris_ProjMatInverse;",
    "iris_NormalMat": "uniform mat3 iris_NormalMat;",
    "iris_TextureMat": "uniform mat4 iris_TextureMat;",
    "iris_LightmapTextureMatrix": "uniform mat4 iris_LightmapTextureMatrix;",
    "iris_RegionOffset": "uniform vec3 iris_RegionOffset;",
}

# Generic vertex attribute locations (bound with glBindAttribLocation before linking)
ATTRIBUTES = {
    "iris_Position": 0, "iris_Color": 1, "iris_UV0": 2, "iris_UV2": 3, "iris_Normal": 4,
    "mc_Entity": 5, "mc_midTexCoord": 6, "at_tangent": 7, "at_midBlock": 8,
    "iris_DhMaterial": 9,   # DH LOD meshes: the DH block material (same data as mc_Entity)
}


def _sub(text, pattern, repl):
    return re.sub(pattern, repl, text)


def patch_world(text, stage):
    """gbuffers_* / shadow* programs: Iris replaces fixed-function inputs with its own."""
    if stage in ("vsh", "gsh", "fsh"):
        text = _sub(text, r"\bgl_ModelViewProjectionMatrix\b", "(iris_ProjMat * iris_ModelViewMat)")
        text = _sub(text, r"\bgl_ModelViewMatrixInverse\b", "iris_ModelViewMatInverse")
        text = _sub(text, r"\bgl_ProjectionMatrixInverse\b", "iris_ProjMatInverse")
        text = _sub(text, r"\bgl_ModelViewMatrix\b", "iris_ModelViewMat")
        text = _sub(text, r"\bgl_ProjectionMatrix\b", "iris_ProjMat")
        text = _sub(text, r"\bgl_NormalMatrix\b", "iris_NormalMat")
        text = _sub(text, r"\bgl_TextureMatrix\s*\[\s*0\s*\]", "iris_TextureMat")
        text = _sub(text, r"\bgl_TextureMatrix\s*\[\s*[12]\s*\]", "iris_LightmapTextureMatrix")
    if stage == "vsh":
        # Sodium: gl_Vertex = position + camera-relative region offset (iris_RegionOffset is 0 in chunk mode)
        text = _sub(text, r"\bftransform\s*\(\s*\)", "iris_ftransform()")
        text = _sub(text, r"\bgl_Vertex\b", "vec4(iris_Position + iris_RegionOffset, 1.0)")
        text = _sub(text, r"\bgl_Normal\b", "iris_Normal")
        text = _sub(text, r"\bgl_Color\b", "iris_Color")
        text = _sub(text, r"\bgl_MultiTexCoord0\b", "vec4(iris_UV0, 0.0, 1.0)")
        text = _sub(text, r"\bgl_MultiTexCoord[12]\b", "vec4(iris_UV2, 0.0, 1.0)")
        text = _sub(text, r"\bgl_MultiTexCoord3\b", "mc_midTexCoord")
        text = _sub(text, r"\bgl_MultiTexCoord[4-7]\b", "vec4(0.0, 0.0, 0.0, 1.0)")
    return text


def patch_composite(text, stage):
    """Fullscreen passes (CompositeTransformer)."""
    text = _sub(text, r"\bgl_TextureMatrix\s*\[\s*\d+\s*\]", "mat4(1.0)")
    text = _sub(text, r"\bgl_Color\b", "vec4(1.0)")
    text = _sub(text, r"\bgl_NormalMatrix\b", "mat3(1.0)")
    ortho = "mat4(vec4(2.0, 0.0, 0.0, 0.0), vec4(0.0, 2.0, 0.0, 0.0), vec4(0.0), vec4(-1.0, -1.0, 0.0, 1.0))"
    text = _sub(text, r"\bgl_ModelViewProjectionMatrix\b", ortho)
    text = _sub(text, r"\bgl_ModelViewMatrix\b", "mat4(1.0)")
    text = _sub(text, r"\bgl_ProjectionMatrix\b", ortho)
    if stage == "vsh":
        text = _sub(text, r"\bftransform\s*\(\s*\)", "iris_ftransform()")
        text = _sub(text, r"\bgl_Vertex\b", "vec4(iris_Position, 1.0)")
        text = _sub(text, r"\bgl_Normal\b", "vec3(0.0, 0.0, 1.0)")
        text = _sub(text, r"\bgl_MultiTexCoord0\b", "vec4(iris_UV0, 0.0, 1.0)")
        text = _sub(text, r"\bgl_MultiTexCoord[1-7]\b", "vec4(0.0, 0.0, 0.0, 1.0)")
    return text


def _injections(text, stage, kind):
    """Declarations that the patched text needs."""
    inj = []
    for name, decl in _MAT_DECLS.items():
        if re.search(r"\b%s\b" % name, text):
            inj.append(decl)
    if stage == "vsh":
        if re.search(r"\biris_Position\b|\biris_ftransform\b", text):
            inj.append("in vec3 iris_Position;")
        for name, decl in (("iris_Color", "in vec4 iris_Color;"), ("iris_UV0", "in vec2 iris_UV0;"),
                           ("iris_UV2", "in vec2 iris_UV2;"), ("iris_Normal", "in vec3 iris_Normal;")):
            if re.search(r"\b%s\b" % name, text):
                inj.append(decl)
        if re.search(r"\biris_ftransform\b", text):
            if kind == "world":
                if "uniform mat4 iris_ModelViewMat;" not in inj:
                    inj.append(_MAT_DECLS["iris_ModelViewMat"])
                if "uniform mat4 iris_ProjMat;" not in inj:
                    inj.append(_MAT_DECLS["iris_ProjMat"])
                if "uniform vec3 iris_RegionOffset;" not in inj:
                    inj.append(_MAT_DECLS["iris_RegionOffset"])
                inj.append("vec4 iris_ftransform() { return iris_ProjMat * (iris_ModelViewMat * "
                           "vec4(iris_Position + iris_RegionOffset, 1.0)); }")
            else:
                inj.append("vec4 iris_ftransform() { return vec4(iris_Position.xy * 2.0 - 1.0, 0.0, 1.0); }")
        if re.search(r"\bmc_midTexCoord\b", text) and not re.search(r"\b(in|attribute)\s+vec[24]\s+mc_midTexCoord\b", text):
            inj.append("in vec4 mc_midTexCoord;")
    return inj


def patch_dh(text, stage):
    """Iris DHTerrainTransformer on top of patch_world: the vertex stage gets `int dhMaterialId`
    (set at the start of main from the per-vertex DH material: attribute iris_DhMaterial, which the
    mesh VAO feeds from the mc_Entity.x float) and
    the fragment stage loses gl_Color (renamed to an undeclared _vert_color, as in Iris).
    gl_TextureMatrix[0/1] = identity and the (level + 0.5) / 16 lightmap encoding come from the
    per-draw uniforms / the DH mesh; gl_ProjectionMatrix is set to the DH projection per draw."""
    if stage == "vsh":
        text, n = re.subn(r"(\bvoid\s+main\s*\(\s*(?:void)?\s*\)\s*\{)",
                          r"\1 dhMaterialId = int(iris_DhMaterial.x + 0.5);", text, count=1)
    elif stage == "fsh":
        text = _sub(text, r"\bgl_Color\b", "_vert_color")
    return text


def build_compile_text(src, stage, kind, macros, dh=False):
    """kind: 'world' (gbuffers/shadow) | 'composite' | 'compute'. dh: a Distant Horizons program."""
    lines = list(src.lines)
    origins = list(src.origins)
    # locate #version (must be first directive)
    vidx = next((i for i, l in enumerate(lines) if l.strip().startswith("#version")), None)
    if vidx is None:
        raise SourceError("no #version line")
    version_line = lines[vidx]
    ext_lines = [(i, l) for i, l in enumerate(lines) if l.strip().startswith("#extension")]
    ext_idx = {i for i, _ in ext_lines}
    body = [(l, origins[i]) for i, l in enumerate(lines) if i != vidx and i not in ext_idx]
    body_text = "\n".join(l for l, _ in body)
    if kind == "world":
        body_text = patch_world(body_text, stage)
        if dh:
            body_text = patch_dh(body_text, stage)
    elif kind == "composite":
        body_text = patch_composite(body_text, stage)
    body_lines = body_text.split("\n")
    assert len(body_lines) == len(body), "patching must not change line count"
    inj = _injections(body_text, stage, kind) if kind != "compute" else []
    if dh and stage == "vsh":
        inj += ["in vec4 iris_DhMaterial;", "int dhMaterialId;"]
    out = [version_line]
    out_orig = [origins[vidx]]
    for i, l in ext_lines:
        out.append(l)
        out_orig.append(origins[i])
    for k, v in macros.items():
        out.append(f"#define {k} {v}".rstrip())
        out_orig.append(("<iris macros>", 0))
    for l in inj:
        out.append(l)
        out_orig.append(("<irisim injected>", 0))
    out.extend(body_lines)
    out_orig.extend(o for _, o in body)
    src.compile_text = "\n".join(out) + "\n"
    src.compile_origins = out_orig
    # active text for directive parsing: standard macros + the source's own defines
    pp = cpp.MiniCpp(macros)
    src.active = pp.process("\n".join(src.lines))
    src.pp_errors = pp.errors
    src.macros_after = pp.macros
    return src


# ---------------------------------------------------------------- directives
class ConstDirective:
    __slots__ = ("type", "key", "value")

    def __init__(self, t, k, v):
        self.type, self.key, self.value = t, k, v


def find_const_directives(text):
    """ConstDirectiveParser.findDirectives (line based, comments irrelevant)."""
    out = []
    for line in text.splitlines():
        if "const" not in line or "=" not in line or ";" not in line:
            continue
        line = line.strip()
        if not line.startswith("const"):
            continue
        rest = line[5:]
        if not rest[:1].isspace():
            continue
        rest = rest.strip()
        typ = None
        for t in ("int", "float", "vec2", "ivec3", "vec4", "bool"):
            if rest.startswith(t):
                typ = t
                rest = rest[len(t):]
                break
        if typ is None or not rest[:1].isspace():
            continue
        eq = rest.find("=")
        if eq < 0:
            continue
        key = rest[:eq].strip()
        if not re.fullmatch(r"\w+", key):
            continue
        remaining = rest[eq + 1:]
        semi = remaining.find(";")
        if semi < 0:
            continue
        out.append(ConstDirective(typ, key, remaining[:semi].strip()))
    return out


def find_comment_directive(text, name):
    """CommentDirectiveParser.findDirective: last occurrence of /* NAME:value */."""
    prefix = name + ":"
    idx = text.rfind(prefix)
    if idx < 0:
        return None, -1
    before = text[:idx].rstrip()
    if not before.endswith("/*"):
        return None, -1
    rest = text[idx + len(prefix):]
    end = rest.find("*/")
    if end < 0:
        return None, -1
    return rest[:end].strip(), idx


def draw_buffers(active_text):
    db, di = find_comment_directive(active_text, "DRAWBUFFERS")
    rt, ri = find_comment_directive(active_text, "RENDERTARGETS")
    if db is not None and rt is not None:
        use_db = di > ri
    else:
        use_db = db is not None
    if use_db:
        return [int(c) for c in db if c.isdigit()], True
    if rt is not None:
        try:
            return [int(x) for x in rt.split(",") if x.strip()], True
        except ValueError:
            return [0], False
    return [0], False


def parse_float(v):
    try:
        return float(v.rstrip("fF"))
    except ValueError:
        return None


def parse_vec(v, n, ctor="vec4"):
    m = re.match(r"\s*[iu]?vec%d\s*\((.*)\)\s*$" % n, v)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    try:
        vals = [float(p.rstrip("fF")) for p in parts]
    except ValueError:
        return None
    if len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        return None
    return vals


def parse_bool(v):
    if v in ("true", "1"):
        return True
    if v in ("false", "0"):
        return False
    return None


# ============================================================================ properties
class PackProperties:
    def __init__(self, items, log):
        self.raw = items
        self.images = []            # dicts
        self.buffer_objects = {}    # index -> size
        self.scale = {}             # pass -> (scale, ox, oy)
        self.size_buffer = {}       # buffer index -> (w spec, h spec)
        self.blend_program = {}     # program -> None (off) or (src, dst, srcA, dstA)
        self.blend_buffer = {}      # program -> {buffer: None | modes}
        self.custom_textures = {}   # sampler -> path or raw spec
        self.flips = {}             # pass -> {buffer: bool}
        self.program_enabled = {}   # program -> expr
        self.uniforms = []          # (kind, type, name, expr) kind uniform|variable
        self.flags = {}             # misc key -> value
        self.noise_texture = None
        for key, value in items:
            try:
                self._handle(key, value, log)
            except Exception as e:  # noqa: BLE001 — never die on a bad property line
                log.warn(f"shaders.properties: cannot parse {key} = {value}: {e}")

    def _handle(self, key, value, log):
        if key == "texture.noise":
            self.noise_texture = value
            return
        if key.startswith("scale."):
            parts = value.split()
            s = float(parts[0])
            ox = float(parts[1]) if len(parts) > 2 else 0.0
            oy = float(parts[2]) if len(parts) > 2 else 0.0
            self.scale[key[6:]] = (s, ox, oy)
            return
        if key.startswith("size.buffer."):
            name = key[len("size.buffer."):]
            m = re.match(r"colortex(\d+)$", name)
            if m:
                parts = value.split()
                self.size_buffer[int(m.group(1))] = (parts[0], parts[1])
            return
        if key.startswith("blend."):
            rest = key[6:]
            if "." in rest:
                prog, buf = rest.split(".", 1)
                legacy = ["gcolor", "gdepth", "gnormal", "composite", "gaux1", "gaux2", "gaux3", "gaux4"]
                idx = legacy.index(buf) if buf in legacy else (int(buf[8:]) if buf.startswith("colortex") else -1)
                if idx < 0:
                    raise ValueError("unknown buffer " + buf)
                self.blend_buffer.setdefault(prog, {})[idx] = None if value == "off" else value.split()
            else:
                self.blend_program[rest] = None if value == "off" else value.split()
            return
        if key.startswith("bufferObject."):
            idx = int(key[len("bufferObject."):])
            parts = value.split()
            size = int(parts[0])
            if size >= 1 and idx <= 12:
                rel = None
                if len(parts) > 2:
                    rel = (parts[1].lower() == "true", float(parts[2]), float(parts[3]))
                self.buffer_objects[idx] = (size, rel)
            return
        if key.startswith("customTexture."):
            self.custom_textures[key[len("customTexture."):]] = value
            return
        if key.startswith("image."):
            name = key[6:]
            p = value.split()
            img = {"name": name, "sampler": None if p[0] == "none" else p[0], "format": p[1], "internal": p[2],
                   "pixel_type": p[3], "clear": p[4].lower() == "true", "relative": p[5].lower() == "true"}
            if img["relative"]:
                img["rel"] = (float(p[6]), float(p[7]))
                img["dims"] = 2
            else:
                sizes = [int(x) for x in p[6:]]
                img["size"] = sizes
                img["dims"] = len(sizes)
            if len(self.images) >= 16:
                log.warn(f"image.{name}: more than 16 images (Iris limit), ignored")
                return
            self.images.append(img)
            return
        if key.startswith("flip."):
            rest = key[5:]
            if "." in rest:
                p, b = rest.rsplit(".", 1)
                m = re.match(r"colortex(\d+)$", b)
                if m:
                    self.flips.setdefault(p, {})[int(m.group(1))] = value.strip().lower() == "true"
            return
        if key.startswith("program.") and key.endswith(".enabled"):
            self.program_enabled[key[len("program."):-len(".enabled")]] = value
            return
        if key.startswith("uniform.") or key.startswith("variable."):
            kind, rest = key.split(".", 1)
            typ, name = rest.split(".", 1)
            self.uniforms.append((kind, typ, name, value))
            return
        self.flags[key] = value

    def flag(self, key, default=None):
        return self.flags.get(key, default)

    def bool_flag(self, key, default):
        v = self.flags.get(key)
        if v is None:
            return default
        return v.strip().lower() in ("true", "on", "1")


def read_properties_file(name, defines):
    path = os.path.join(SHADERS, name)
    if not os.path.isfile(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out, pp = cpp.preprocess_properties(text, defines)
    return cpp.parse_properties(out), pp


# ============================================================================ id maps
class IdMap:
    """block/item/entity .properties -> numeric ids."""

    def __init__(self, items, prefix):
        self.entries = []  # (id, name, props dict)
        self.layers = {}   # block name -> render layer from Iris "layer.<type> = <blocks>" lines (block.properties)
        if not items:
            return
        for key, value in items:
            if prefix == "block" and key.startswith("layer."):
                for entry in value.split():
                    n = entry[len("minecraft:"):] if entry.startswith("minecraft:") else entry
                    self.layers[n] = key[len("layer."):]
                continue
            if not key.startswith(prefix + "."):
                continue
            try:
                ident = int(key[len(prefix) + 1:])
            except ValueError:
                continue
            for entry in value.split():
                if entry.startswith("%"):
                    continue  # tags not supported
                parts = entry.split(":")
                if parts and parts[0] == "minecraft":
                    parts = parts[1:]
                elif len(parts) > 1 and "=" not in parts[1]:
                    # other namespace: namespace:name[:props]
                    parts = [parts[0] + ":" + parts[1]] + parts[2:]
                if not parts:
                    continue
                name = parts[0]
                props = {}
                for p in parts[1:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        props[k] = v.split(",")
                self.entries.append((ident, name, props))

    def lookup(self, name, state=None, default=-1):
        state = state or {}
        for ident, n, props in self.entries:
            if n != name:
                continue
            ok = True
            for k, vals in props.items():
                if k in state and str(state[k]) not in vals:
                    ok = False
                    break
            if ok:
                return ident
        return default

    def __len__(self):
        return len(self.entries)


# ============================================================================ program discovery
def program_file(world, name, ext):
    for d in (os.path.join(SHADERS, world), SHADERS):
        p = os.path.join(d, f"{name}.{ext}")
        if os.path.isfile(p):
            return p
    return None


def compute_array(world, name, enabled=None, dropped=None):
    """Iris ProgramSet.readComputeArray: name.csh, then name_a.csh .. name_z.csh, stopping at the first
    missing letter. Returns [(program name, path)].

    `enabled(program name) -> bool` applies `program.<world>/<name>.enabled`: Iris' source provider returns
    null for a disabled program (ShaderPack: disabledPrograms), so a disabled program is *missing* to
    readComputeArray. A disabled base `name.csh` is simply skipped (the letters are read whatever entry 0
    is), but a disabled letter ends the array like a missing file: every later letter is dropped too, even
    when it exists and is enabled. Such dropped letters (existing and enabled, but behind a disabled or
    missing letter) are appended to `dropped` when a list is given."""
    ok = enabled or (lambda n: True)
    out = []
    p = program_file(world, name, "csh")
    if p and ok(name):
        out.append((name, p))
    ended = False
    for c in "abcdefghijklmnopqrstuvwxyz":
        sub = f"{name}_{c}"
        q = program_file(world, sub, "csh")
        if ended:
            if q and dropped is not None and ok(sub):
                dropped.append(sub)
            continue
        if not q or not ok(sub):
            ended = True
            continue
        out.append((sub, q))
    return out


def has_program(world, name):
    return program_file(world, name, "vsh") is not None and program_file(world, name, "fsh") is not None
