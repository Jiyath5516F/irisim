"""Thin OpenGL helpers: Iris texture formats, textures, sampler objects, programs."""
import ctypes
import re

import numpy as np
from OpenGL import GL

from . import pack

# ---------------------------------------------------------------- formats
def format_info(name):
    """Iris InternalTextureFormat name -> dict(internal, fmt, type, kind, channels)."""
    n = name.upper()
    if n == "RGBA":
        n = "RGBA8"
    if n == "RGB":
        n = "RGB8"
    internal = getattr(GL, "GL_" + n, None)
    if internal is None:
        return None
    m = re.match(r"(RGBA|RGB|RG|R)(\d+)?(.*)$", n)
    special = {"R11F_G11F_B10F": ("float", 3), "RGB10_A2": ("unorm", 4), "RGB10_A2UI": ("uint", 4),
               "RGB5_A1": ("unorm", 4), "RGB9_E5": ("float", 3), "RGB565": ("unorm", 3)}
    if n in special:
        kind, ch = special[n]
    elif m:
        ch = len(m.group(1))
        suffix = m.group(3)
        if suffix == "F":
            kind = "float"
        elif suffix == "UI":
            kind = "uint"
        elif suffix == "I":
            kind = "int"
        elif suffix == "_SNORM":
            kind = "snorm"
        else:
            kind = "unorm"
    else:
        return None
    base = {1: "RED", 2: "RG", 3: "RGB", 4: "RGBA"}[ch]
    if kind in ("int", "uint"):
        fmt = getattr(GL, f"GL_{base}_INTEGER")
        typ = GL.GL_UNSIGNED_INT if kind == "uint" else GL.GL_INT
    else:
        fmt = getattr(GL, f"GL_{base}")
        typ = GL.GL_FLOAT
    return {"name": n, "internal": internal, "fmt": fmt, "type": typ, "kind": kind, "channels": ch}


IMAGE_FORMATS_OK = {"RGBA32F", "RGBA16F", "RG32F", "RG16F", "R11F_G11F_B10F", "R32F", "R16F", "RGBA32UI", "RGBA16UI",
                    "RGB10_A2UI", "RGBA8UI", "RG32UI", "RG16UI", "RG8UI", "R32UI", "R16UI", "R8UI", "RGBA32I",
                    "RGBA16I", "RGBA8I", "RG32I", "RG16I", "RG8I", "R32I", "R16I", "R8I", "RGBA16", "RGB10_A2",
                    "RGBA8", "RG16", "RG8", "R16", "R8", "RGBA16_SNORM", "RGBA8_SNORM", "RG16_SNORM", "RG8_SNORM",
                    "R16_SNORM", "R8_SNORM"}


class Texture:
    _count = 0

    def __init__(self, target, fmt_name, w, h=1, d=1, label=None, depth=False, data=None, data_fmt=None,
                 data_type=None):
        self.target = target
        self.w, self.h, self.d = w, h, d
        self.label = label
        self.depth = depth
        self.id = int(GL.glGenTextures(1))
        GL.glBindTexture(target, self.id)
        if depth:
            self.info = {"name": fmt_name, "internal": getattr(GL, "GL_" + fmt_name), "fmt": GL.GL_DEPTH_COMPONENT,
                         "type": GL.GL_FLOAT, "kind": "depth", "channels": 1}
        else:
            self.info = format_info(fmt_name)
            if self.info is None:
                raise ValueError(f"unsupported texture format {fmt_name}")
        info = self.info
        dfmt = data_fmt if data_fmt is not None else info["fmt"]
        dtyp = data_type if data_type is not None else info["type"]
        if isinstance(data, (bytes, bytearray)):
            # raw file contents: hand the bytes to GL unchanged (PyOpenGL's array wrappers would convert a
            # numpy array to the element type of `dtyp`), through the raw entry points
            from OpenGL.raw.GL.VERSION.GL_1_0 import glTexImage1D as rawTexImage1D, glTexImage2D as rawTexImage2D
            from OpenGL.raw.GL.VERSION.GL_1_2 import glTexImage3D as rawTexImage3D
            buf = (ctypes.c_ubyte * max(1, len(data))).from_buffer_copy(bytes(data) or b"\0")
            ptr = ctypes.cast(buf, ctypes.c_void_p)
            if target in (GL.GL_TEXTURE_2D, GL.GL_TEXTURE_RECTANGLE):
                rawTexImage2D(target, 0, info["internal"], w, h, 0, dfmt, dtyp, ptr)
            elif target == GL.GL_TEXTURE_3D:
                rawTexImage3D(target, 0, info["internal"], w, h, d, 0, dfmt, dtyp, ptr)
            elif target == GL.GL_TEXTURE_1D:
                rawTexImage1D(target, 0, info["internal"], w, 0, dfmt, dtyp, ptr)
            else:
                raise ValueError("unsupported target")
            del buf
        elif target in (GL.GL_TEXTURE_2D, GL.GL_TEXTURE_RECTANGLE):
            GL.glTexImage2D(target, 0, info["internal"], w, h, 0, dfmt, dtyp, data)
        elif target == GL.GL_TEXTURE_3D:
            GL.glTexImage3D(target, 0, info["internal"], w, h, d, 0, dfmt, dtyp, data)
        elif target == GL.GL_TEXTURE_1D:
            GL.glTexImage1D(target, 0, info["internal"], w, 0, dfmt, dtyp, data)
        else:
            raise ValueError("unsupported target")
        # textures are always sampled through sampler objects; keep the texture itself complete
        GL.glTexParameteri(target, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(target, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glBindTexture(target, 0)
        if label:
            try:
                GL.glObjectLabel(GL.GL_TEXTURE, self.id, len(label), label.encode())
            except Exception:  # noqa: BLE001
                pass

    @property
    def kind(self):
        return self.info["kind"]

    @property
    def is_integer(self):
        return self.info["kind"] in ("int", "uint")

    def clear(self, color):
        """glClearTexImage with a 4-component colour."""
        if self.depth:
            GL.glClearTexImage(self.id, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, np.array([color[0]], np.float32))
        elif self.kind == "uint":
            GL.glClearTexImage(self.id, 0, GL.GL_RGBA_INTEGER, GL.GL_UNSIGNED_INT,
                               np.array([int(max(0, c)) for c in color], np.uint32))
        elif self.kind == "int":
            GL.glClearTexImage(self.id, 0, GL.GL_RGBA_INTEGER, GL.GL_INT, np.array([int(c) for c in color], np.int32))
        else:
            GL.glClearTexImage(self.id, 0, GL.GL_RGBA, GL.GL_FLOAT, np.array(color, np.float32))

    def read(self, level=0):
        """Return numpy array (d, h, w, 4) or (h, w, 4) / depth (h, w)."""
        w = max(1, self.w >> level)
        h = max(1, self.h >> level)
        d = max(1, self.d >> level) if self.target == GL.GL_TEXTURE_3D else 1
        if self.depth:
            a = np.zeros((d * h * w,), np.float32)
            GL.glGetTextureImage(self.id, level, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, a.nbytes, a)
            return a.reshape(h, w)
        ch = self.info["channels"]
        ch = 4 if ch == 3 else ch
        base = {1: "RED", 2: "RG", 4: "RGBA"}[ch]
        if self.kind == "uint":
            a = np.zeros((d, h, w, ch), np.uint32)
            GL.glGetTextureImage(self.id, level, getattr(GL, f"GL_{base}_INTEGER"), GL.GL_UNSIGNED_INT, a.nbytes, a)
        elif self.kind == "int":
            a = np.zeros((d, h, w, ch), np.int32)
            GL.glGetTextureImage(self.id, level, getattr(GL, f"GL_{base}_INTEGER"), GL.GL_INT, a.nbytes, a)
        else:
            a = np.zeros((d, h, w, ch), np.float32)
            GL.glGetTextureImage(self.id, level, getattr(GL, f"GL_{base}"), GL.GL_FLOAT, a.nbytes, a)
        return a if self.target == GL.GL_TEXTURE_3D else a[0]

    def read_rgba(self, level=0):
        """Raw read-back for numeric tests (`--npy`): always 4 channels in GL row order (row 0 = bottom,
        no flip), (d, h, w, 4) for 3D textures, (h, w, 4) otherwise. float32 for float / unorm / snorm
        formats (GL's RGBA expansion of formats with fewer channels: missing colour channels 0, alpha 1),
        uint32 / int32 for integer formats (unconverted), (h, w) float32 for depth textures."""
        w = max(1, self.w >> level)
        h = max(1, self.h >> level)
        d = max(1, self.d >> level) if self.target == GL.GL_TEXTURE_3D else 1
        if self.depth:
            return self.read(level)
        if self.kind in ("uint", "int"):
            dt, gt = (np.uint32, GL.GL_UNSIGNED_INT) if self.kind == "uint" else (np.int32, GL.GL_INT)
            a = np.zeros((d, h, w, 4), dt)
            GL.glGetTextureImage(self.id, level, GL.GL_RGBA_INTEGER, gt, a.nbytes, a)
        else:
            a = np.zeros((d, h, w, 4), np.float32)
            GL.glGetTextureImage(self.id, level, GL.GL_RGBA, GL.GL_FLOAT, a.nbytes, a)
        return a if self.target == GL.GL_TEXTURE_3D else a[0]

    def generate_mipmaps(self):
        GL.glGenerateTextureMipmap(self.id)

    def delete(self):
        GL.glDeleteTextures([self.id])


class Samplers:
    """Cache of GL sampler objects."""

    def __init__(self):
        self.cache = {}

    def get(self, min_f="linear", mag_f="linear", wrap="clamp", compare=False, wrap_r=None):
        """wrap: S and T ("clamp" | "repeat"); wrap_r: R, the same as `wrap` unless given (Iris' GlSampler
        never sets WRAP_R, so its custom-texture samplers keep the GL default REPEAT on R)."""
        wrap_r = wrap if wrap_r is None else wrap_r
        key = (min_f, mag_f, wrap, compare, wrap_r)
        if key in self.cache:
            return self.cache[key]
        s = int(GL.glGenSamplers(1))
        mins = {"linear": GL.GL_LINEAR, "nearest": GL.GL_NEAREST, "linear_mip": GL.GL_LINEAR_MIPMAP_LINEAR,
                "nearest_mip_linear": GL.GL_NEAREST_MIPMAP_LINEAR, "nearest_mip": GL.GL_NEAREST_MIPMAP_NEAREST}
        GL.glSamplerParameteri(s, GL.GL_TEXTURE_MIN_FILTER, mins[min_f])
        GL.glSamplerParameteri(s, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR if mag_f == "linear" else GL.GL_NEAREST)
        w = GL.GL_CLAMP_TO_EDGE if wrap == "clamp" else GL.GL_REPEAT
        for p in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
            GL.glSamplerParameteri(s, p, w)
        GL.glSamplerParameteri(s, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE if wrap_r == "clamp" else GL.GL_REPEAT)
        if compare:
            GL.glSamplerParameteri(s, GL.GL_TEXTURE_COMPARE_MODE, GL.GL_COMPARE_REF_TO_TEXTURE)
            GL.glSamplerParameteri(s, GL.GL_TEXTURE_COMPARE_FUNC, GL.GL_LEQUAL)
        self.cache[key] = s
        return s


# ---------------------------------------------------------------- GLSL type tables
SAMPLER_TYPES = {}
IMAGE_TYPES = {}


def _init_types():
    for n in dir(GL):
        if not n.startswith("GL_"):
            continue
        short = n[3:]
        val = getattr(GL, n)
        if not isinstance(val, int):
            try:
                val = int(val)
            except (TypeError, ValueError):
                continue
        if re.match(r"(INT_|UNSIGNED_INT_)?SAMPLER_", short):
            SAMPLER_TYPES[val] = short
        elif re.match(r"(INT_|UNSIGNED_INT_)?IMAGE_(1D|2D|3D|CUBE|BUFFER|2D_ARRAY|1D_ARRAY|2D_RECT|2D_MULTISAMPLE)",
                      short):
            IMAGE_TYPES[val] = short


_init_types()


def sampler_target(type_name):
    """'UNSIGNED_INT_SAMPLER_3D' -> (GL target, kind)."""
    kind = "uint" if type_name.startswith("UNSIGNED_INT_") else ("int" if type_name.startswith("INT_") else "float")
    core = type_name.split("SAMPLER_", 1)[1]
    shadow = core.endswith("_SHADOW")
    core = core.replace("_SHADOW", "")
    target = {"1D": GL.GL_TEXTURE_1D, "2D": GL.GL_TEXTURE_2D, "3D": GL.GL_TEXTURE_3D, "CUBE": GL.GL_TEXTURE_CUBE_MAP,
              "2D_ARRAY": GL.GL_TEXTURE_2D_ARRAY, "1D_ARRAY": GL.GL_TEXTURE_1D_ARRAY,
              "2D_RECT": GL.GL_TEXTURE_RECTANGLE, "BUFFER": GL.GL_TEXTURE_BUFFER}.get(core, GL.GL_TEXTURE_2D)
    return target, kind, shadow


FLOAT_T = {GL.GL_FLOAT: 1, GL.GL_FLOAT_VEC2: 2, GL.GL_FLOAT_VEC3: 3, GL.GL_FLOAT_VEC4: 4}
INT_T = {GL.GL_INT: 1, GL.GL_INT_VEC2: 2, GL.GL_INT_VEC3: 3, GL.GL_INT_VEC4: 4, GL.GL_BOOL: 1, GL.GL_BOOL_VEC2: 2,
         GL.GL_BOOL_VEC3: 3, GL.GL_BOOL_VEC4: 4}
UINT_T = {GL.GL_UNSIGNED_INT: 1, GL.GL_UNSIGNED_INT_VEC2: 2, GL.GL_UNSIGNED_INT_VEC3: 3, GL.GL_UNSIGNED_INT_VEC4: 4}
MAT_T = {GL.GL_FLOAT_MAT2: 2, GL.GL_FLOAT_MAT3: 3, GL.GL_FLOAT_MAT4: 4}


def set_uniform(loc, gltype, value):
    """Upload a python/numpy value according to the declared GLSL type."""
    if gltype in FLOAT_T:
        n = FLOAT_T[gltype]
        v = np.atleast_1d(np.asarray(value, np.float64)).astype(np.float32)
        if v.size < n:
            v = np.concatenate([v, np.zeros(n - v.size, np.float32)])
        (GL.glUniform1f, GL.glUniform2f, GL.glUniform3f, GL.glUniform4f)[n - 1](loc, *[float(x) for x in v[:n]])
    elif gltype in INT_T:
        n = INT_T[gltype]
        v = np.atleast_1d(np.asarray(value)).astype(np.float64)
        if v.size < n:
            v = np.concatenate([v, np.zeros(n - v.size)])
        iv = [int(np.floor(x)) if gltype not in (GL.GL_BOOL,) else int(bool(x)) for x in v[:n]]
        (GL.glUniform1i, GL.glUniform2i, GL.glUniform3i, GL.glUniform4i)[n - 1](loc, *iv)
    elif gltype in UINT_T:
        n = UINT_T[gltype]
        v = np.atleast_1d(np.asarray(value)).astype(np.float64)
        if v.size < n:
            v = np.concatenate([v, np.zeros(n - v.size)])
        (GL.glUniform1ui, GL.glUniform2ui, GL.glUniform3ui, GL.glUniform4ui)[n - 1](loc, *[int(max(0, x)) for x in v[:n]])
    elif gltype in MAT_T:
        n = MAT_T[gltype]
        m = np.asarray(value, np.float64)
        if m.shape != (n, n):
            m4 = np.eye(4)
            k = min(4, m.shape[0])
            m4[:k, :k] = m[:k, :k]
            m = m4[:n, :n]
        f = (None, None, GL.glUniformMatrix2fv, GL.glUniformMatrix3fv, GL.glUniformMatrix4fv)[n]
        f(loc, 1, GL.GL_TRUE, np.ascontiguousarray(m, np.float32))
    else:
        return False
    return True


class Program:
    """A linked GL program with Iris-relevant introspection."""

    def __init__(self, name, stages, log, kind):
        """stages: dict ext -> pack.Source (with compile_text)."""
        self.name = name
        self.kind = kind
        self.ok = False
        self.errors = []
        self.uniforms = {}     # name -> (loc, type)
        self.samplers = []     # (name, loc, type_name, unit)
        self.images = []       # (name, loc, type_name, unit)
        self.local_size = (1, 1, 1)
        self.id = 0
        self.stages = stages
        shader_types = {"vsh": GL.GL_VERTEX_SHADER, "fsh": GL.GL_FRAGMENT_SHADER, "gsh": GL.GL_GEOMETRY_SHADER,
                        "csh": GL.GL_COMPUTE_SHADER}
        shaders = []
        for ext, src in stages.items():
            sh = GL.glCreateShader(shader_types[ext])
            GL.glShaderSource(sh, src.compile_text)
            GL.glCompileShader(sh)
            if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
                msg = GL.glGetShaderInfoLog(sh)
                msg = msg.decode(errors="replace") if isinstance(msg, bytes) else str(msg)
                self.errors.append(f"[{ext}] " + self._map_lines(msg, src))
            shaders.append(sh)
        if self.errors:
            for sh in shaders:
                GL.glDeleteShader(sh)
            return
        prog = GL.glCreateProgram()
        for sh in shaders:
            GL.glAttachShader(prog, sh)
        for aname, aloc in pack.ATTRIBUTES.items():
            GL.glBindAttribLocation(prog, aloc, aname)
        GL.glLinkProgram(prog)
        for sh in shaders:
            GL.glDetachShader(prog, sh)
            GL.glDeleteShader(sh)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            msg = GL.glGetProgramInfoLog(prog)
            msg = msg.decode(errors="replace") if isinstance(msg, bytes) else str(msg)
            self.errors.append("[link] " + msg.strip())
            GL.glDeleteProgram(prog)
            return
        self.id = int(prog)
        self.ok = True
        self._introspect()
        if "csh" in stages:
            ls = np.zeros(3, np.int32)
            GL.glGetProgramiv(self.id, GL.GL_COMPUTE_WORK_GROUP_SIZE, ls)
            self.local_size = tuple(int(x) for x in ls)

    @staticmethod
    def _map_lines(msg, src):
        out = []
        for line in msg.strip().splitlines():
            m = re.match(r"\s*(\d+):(\d+)\((\d+)\):(.*)", line)
            if m:
                ln = int(m.group(2))
                out.append(f"{src.origin(ln)}: {m.group(4).strip()}")
            else:
                out.append(line.strip())
        return "\n    ".join(out)

    def _introspect(self):
        GL.glUseProgram(self.id)
        n = GL.glGetProgramiv(self.id, GL.GL_ACTIVE_UNIFORMS)
        sunit = 0
        iunit = 0
        for i in range(n):
            name, size, typ = GL.glGetActiveUniform(self.id, i)
            name = name.decode() if isinstance(name, bytes) else str(name)
            typ = int(typ)
            loc = GL.glGetUniformLocation(self.id, name)
            if loc < 0:
                continue  # block member
            base = name[:-3] if name.endswith("[0]") else name
            if typ in SAMPLER_TYPES:
                GL.glUniform1i(loc, sunit)
                self.samplers.append((base, loc, SAMPLER_TYPES[typ], sunit))
                sunit += 1
            elif typ in IMAGE_TYPES:
                GL.glUniform1i(loc, iunit)
                self.images.append((base, loc, IMAGE_TYPES[typ], iunit))
                iunit += 1
            else:
                self.uniforms[base] = (loc, typ)
        GL.glUseProgram(0)

    def use(self):
        GL.glUseProgram(self.id)
