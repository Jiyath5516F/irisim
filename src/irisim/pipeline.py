"""The mini-Iris rendering pipeline.

Mirrors net.irisshaders.iris.pipeline.IrisRenderingPipeline / CompositeRenderer /
FinalPassRenderer / ShadowRenderer semantics on top of raw OpenGL (PyOpenGL)."""
import ctypes
import math
import os
import time

import numpy as np
from OpenGL import GL
from PIL import Image

from . import dumps
from . import glutil
from . import pack
from . import scene

LEGACY_TARGETS = ["gcolor", "gdepth", "gnormal", "composite", "gaux1", "gaux2", "gaux3", "gaux4"]

# Iris ProgramId order (gbuffer + shadow programs) — used for directive collection order
PROGRAM_ID_ORDER = ["shadow", "shadow_solid", "shadow_cutout", "shadow_water", "shadow_entities", "shadow_lightning",
                    "shadow_block", "gbuffers_basic", "gbuffers_line", "gbuffers_textured", "gbuffers_textured_lit",
                    "gbuffers_skybasic", "gbuffers_skytextured", "gbuffers_clouds", "gbuffers_terrain",
                    "gbuffers_terrain_solid", "gbuffers_terrain_cutout", "gbuffers_damagedblock", "gbuffers_block",
                    "gbuffers_block_translucent", "gbuffers_beaconbeam", "gbuffers_item", "gbuffers_entities",
                    "gbuffers_entities_translucent", "gbuffers_lightning", "gbuffers_particles",
                    "gbuffers_particles_translucent", "gbuffers_entities_glowing", "gbuffers_armor_glint",
                    "gbuffers_spidereyes", "gbuffers_hand", "gbuffers_weather", "gbuffers_water",
                    "gbuffers_hand_water"]

BLEND_FUNCS = {n: getattr(GL, "GL_" + n) for n in
               ["ZERO", "ONE", "SRC_COLOR", "ONE_MINUS_SRC_COLOR", "DST_COLOR", "ONE_MINUS_DST_COLOR", "SRC_ALPHA",
                "ONE_MINUS_SRC_ALPHA", "DST_ALPHA", "ONE_MINUS_DST_ALPHA", "SRC_ALPHA_SATURATE"]}
TRANSLUCENT_BLEND = (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)

FALLBACK_VSH = """#version 450 compatibility
out vec2 uv; out vec4 col; out vec2 lm;
void main() { gl_Position = ftransform(); uv = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lm = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy; col = gl_Color; }
"""
FALLBACK_FSH = """#version 450 compatibility
uniform sampler2D gtexture; uniform sampler2D lightmap;
in vec2 uv; in vec4 col; in vec2 lm;
/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 o;
void main() { vec4 c = texture(gtexture, uv) * col; if (c.a < 0.1) discard; o = c * texture(lightmap, lm); }
"""


class Log:
    def __init__(self, path, quiet=False):
        self.f = open(path, "w", encoding="utf-8")
        self.quiet = quiet
        self.counts = {"warn": 0, "error": 0}
        self._once = set()

    def _w(self, s, echo=True):
        self.f.write(s + "\n")
        self.f.flush()
        if echo and not self.quiet:
            print(s, flush=True)

    def info(self, s, echo=True):
        self._w(s, echo)

    def warn(self, s):
        self.counts["warn"] += 1
        self._w("WARN  " + s)

    def error(self, s):
        self.counts["error"] += 1
        self._w("ERROR " + s)

    def once(self, key, s):
        if key not in self._once:
            self._once.add(key)
            self._w(s, echo=False)

    def section(self, s):
        self._w("\n== " + s)


class ProgInfo:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind
        self.stages = {}
        self.prog = None
        self.draw_buffers = [0]
        self.known_db = False
        self.consts = []
        self.compile_ms = 0.0
        self.error = None
        self.work_groups = None
        self.work_groups_rel = None

    @property
    def ok(self):
        return self.prog is not None and self.prog.ok

    def const(self, key, typ=None):
        v = None
        for d in self.consts:
            if d.key == key and (typ is None or d.type == typ):
                v = d.value
        return v


class Pass:
    def __init__(self, group, index, name):
        self.group = group
        self.index = index
        self.name = name
        self.frag = None
        self.computes = []
        self.reads_alt = frozenset()
        self.writes = []
        self.mip = set()
        self.scale = (1.0, 0.0, 0.0)
        self.fbo = None
        self.blend = "default"


class RenderTarget:
    def __init__(self, idx, fmt, w, h):
        self.idx = idx
        self.fmt = fmt
        self.w, self.h = w, h
        self.main = glutil.Texture(GL.GL_TEXTURE_2D, fmt, w, h, label=f"colortex{idx}")
        self.alt = glutil.Texture(GL.GL_TEXTURE_2D, fmt, w, h, label=f"colortex{idx}_alt")


class Mesh:
    def __init__(self, verts, sortable=False):
        self.n_quads = len(verts) // 4
        self.vao = int(GL.glGenVertexArrays(1))
        self.vbo = int(GL.glGenBuffers(1))
        self.ibo = int(GL.glGenBuffers(1))
        self.centers = verts[:, 0:3].reshape(-1, 4, 3).mean(axis=1) if sortable and self.n_quads else None
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        if self.n_quads:
            GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, np.ascontiguousarray(verts, np.float32), GL.GL_STATIC_DRAW)
        stride = scene.VERTEX_FLOATS * 4
        layout = [(0, 3, 0), (1, 4, 3), (2, 2, 7), (3, 2, 9), (4, 3, 11), (5, 2, 14), (6, 2, 16), (7, 4, 18), (8, 4, 22),
                  (9, 1, 14)]   # 9: iris_DhMaterial (DH LOD meshes keep the DH material in mc_Entity.x)
        for loc, n, off in layout:
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, n, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(off * 4))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        self.set_order(np.arange(self.n_quads))
        GL.glBindVertexArray(0)

    def update_vertices(self, verts):
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, verts.nbytes, np.ascontiguousarray(verts, np.float32))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def set_order(self, order):
        base = (np.asarray(order, np.uint32) * 4)[:, None]
        idx = (base + np.array([0, 1, 2, 2, 3, 0], np.uint32)[None, :]).reshape(-1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        if idx.size:
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_DYNAMIC_DRAW)
        self.count = int(idx.size)

    def sort_back_to_front(self, cam_region):
        if self.centers is None or not self.n_quads:
            return
        d = np.sum((self.centers - cam_region[None, :]) ** 2, axis=1)
        GL.glBindVertexArray(self.vao)
        self.set_order(np.argsort(-d, kind="stable"))
        GL.glBindVertexArray(0)

    def draw(self):
        if self.count == 0:
            return
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self.count, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)


class Pipeline:
    def __init__(self, cfg, log, out_dir):
        self.cfg = cfg
        self.log = log
        self.out_dir = out_dir
        self.world = cfg.world
        self.w, self.h = cfg.width, cfg.height
        self.timings = {}
        self.mip_enabled = set()
        self.missing_uniforms = {}
        self.missing_samplers = {}
        self.failed = []
        self.fbo_cache = {}
        self.samplers = glutil.Samplers()
        self.dumper = None
        self.frame_dump = False
        self.frame_index = 0          # set by cli.py before each frame

        log.section("pack")
        self.opts = pack.Options(cfg.sets, cfg.offs, cfg.ons)
        for u in self.opts.unknown:
            log.warn(f"option {u} is not declared by the pack (the override only rewrites lines that declare it)")
        if self.opts.overrides:
            log.info("option overrides: " + ", ".join(f"{k}={v}" for k, v in sorted(self.opts.overrides.items())))
        self.dh = bool(getattr(cfg, "dh", False))
        self.macros = pack.standard_macro_dict(self.dh)
        env = dict(self.macros)
        env.update(self.opts.macros())
        self.prop_env = env
        items, pp = pack.read_properties_file("shaders.properties", env)
        if items is None:
            log.warn("shaders/shaders.properties not found")
            items = []
        elif pp.errors:
            for e in pp.errors:
                log.warn("shaders.properties preprocessor: " + e)
        self.props = pack.PackProperties(items, log)
        self.ids = {}
        for fname, prefix in (("block.properties", "block"), ("item.properties", "item"),
                              ("entity.properties", "entity")):
            it, _ = pack.read_properties_file(fname, env)
            self.ids[prefix] = pack.IdMap(it, prefix)
            log.info(f"{fname}: {'missing' if it is None else str(len(self.ids[prefix])) + ' entries'}")

        from . import customuniforms
        self.custom = customuniforms.CustomUniforms(self.props.uniforms, log)
        if self.props.uniforms:
            log.info(f"custom uniforms/variables: {len(self.custom.items)}")
        self._load_programs()
        self._collect_directives()
        self._build_schedule()

    def custom_uniforms(self, U):
        return self.custom.evaluate(U)

    # ================================================================== loading
    def _enabled(self, name):
        expr = self.props.program_enabled.get(f"{self.world}/{name}", self.props.program_enabled.get(name))
        if expr is None:
            return True
        from .cpp import MiniCpp, CppError
        try:
            return MiniCpp(self.prop_env).eval_condition(expr)
        except CppError as e:
            self.log.warn(f"program.{name}.enabled = {expr}: {e}")
            return True

    def _compute_array(self, name):
        """Compute programs of one pass (`name.csh`, `name_a..z.csh`) with Iris' disabling semantics
        (pack.compute_array): a disabled sub-pass also drops every later letter, which is logged."""
        dropped = []
        progs = pack.compute_array(self.world, name, self._enabled, dropped)
        if dropped:
            self.log.warn(f"{name}: compute sub-passes {', '.join(dropped)} are enabled but not run: Iris stops "
                          f"reading {name}_a..z at the first missing or disabled letter (ProgramSet.readComputeArray)")
        return progs

    def _load(self, name, kind):
        info = ProgInfo(name, kind)
        exts = ["csh"] if kind == "compute" else ["vsh", "gsh", "fsh"]
        try:
            for ext in exts:
                p = pack.program_file(self.world, name, ext)
                if p is None:
                    continue
                src = pack.load_stage(p, self.opts.overrides)
                pack.build_compile_text(src, ext, kind, self.macros, dh=name.startswith("dh_"))
                for e in getattr(src, "pp_errors", []):
                    self.log.warn(f"{name}.{ext}: preprocessor: {e}")
                info.stages[ext] = src
        except pack.SourceError as e:
            info.error = str(e)
            self.log.error(f"{name}: {e}")
            self.failed.append(name)
            return info
        if "fsh" in info.stages:
            act = info.stages["fsh"].active
            info.draw_buffers, info.known_db = pack.draw_buffers(act)
            limit = 8 if (kind == "world" and name.startswith("shadow")) else 16
            bad = [b for b in info.draw_buffers if not 0 <= b < limit]
            if bad:
                self.log.warn(f"{name}: RENDERTARGETS {info.draw_buffers} — ignoring out-of-range {bad}")
                info.draw_buffers = [b for b in info.draw_buffers if 0 <= b < limit] or [0]
            if len(info.draw_buffers) > 8:
                self.log.warn(f"{name}: {len(info.draw_buffers)} draw buffers > GL_MAX_DRAW_BUFFERS (8)")
            info.consts = pack.find_const_directives(act)
        if "csh" in info.stages:
            info.consts = pack.find_const_directives(info.stages["csh"].active)
            wg = info.const("workGroups", "ivec3")
            if wg is not None:
                v = pack.parse_vec(wg, 3)
                if v is None:
                    self.log.warn(f"{name}: cannot parse workGroups = {wg}")
                else:
                    info.work_groups = tuple(int(x) for x in v)
            wr = info.const("workGroupsRender", "vec2")
            if wr is not None:
                v = pack.parse_vec(wr, 2)
                if v is None:
                    self.log.warn(f"{name}: cannot parse workGroupsRender = {wr}")
                else:
                    info.work_groups_rel = tuple(v)
        t0 = time.perf_counter()
        info.prog = glutil.Program(name, info.stages, self.log, kind)
        info.compile_ms = (time.perf_counter() - t0) * 1000
        if not info.prog.ok:
            self.failed.append(name)
            self.log.error(f"{name}: compile/link failed")
            for e in info.prog.errors:
                for line in e.splitlines()[:40]:
                    self.log.info("    " + line)
            dump_dir = os.path.join(self.out_dir, "failed_sources")
            os.makedirs(dump_dir, exist_ok=True)
            for ext, src in info.stages.items():
                with open(os.path.join(dump_dir, f"{name}.{ext}.glsl"), "w") as f:
                    for i, l in enumerate(src.compile_text.split("\n"), 1):
                        f.write(f"{i:5d}  {l}\n")
        else:
            self.log.info(f"  compiled {name:32s} {info.compile_ms:7.0f} ms"
                          + (f"  RT {info.draw_buffers}" if "fsh" in info.stages else "")
                          + (f"  local {info.prog.local_size} groups {info.work_groups or info.work_groups_rel or 'default'}"
                             if kind == "compute" else ""), echo=self.cfg.verbose)
        return info

    def _load_programs(self):
        log = self.log
        log.section(f"programs ({self.world})")
        t0 = time.perf_counter()
        W = self.world
        self.setup = []
        for i in range(100):
            name = "setup" + ("" if i == 0 else str(i))
            if pack.program_file(W, name, "csh") and self._enabled(name):
                self.setup.append(self._load(name, "compute"))
        self.shadow_computes = [self._load(n, "compute") for n, _ in self._compute_array("shadow")]
        self.groups = {}
        for g in pack.COMPOSITE_GROUPS:
            passes = []
            for i in range(100):
                name = g + ("" if i == 0 else str(i))
                has_frag = pack.has_program(W, name) and self._enabled(name)
                comps = self._compute_array(name)
                if not has_frag and not comps:
                    continue
                ps = Pass(g, i, name)
                if has_frag:
                    ps.frag = self._load(name, "composite")
                ps.computes = [self._load(n, "compute") for n, _ in comps]
                passes.append(ps)
            self.groups[g] = passes
        self.final = Pass("final", 0, "final")
        if pack.has_program(W, "final") and self._enabled("final"):
            self.final.frag = self._load("final", "composite")
        self.final.computes = [self._load(n, "compute") for n, _ in self._compute_array("final")]
        self.world_progs = {}
        for name in PROGRAM_ID_ORDER:
            if pack.has_program(W, name) and self._enabled(name):
                self.world_progs[name] = self._load(name, "world")
        # Distant Horizons programs (Iris compiles them only while DH renders)
        self.dh_progs = {}
        if self.dh:
            for name in pack.DH_PROGRAMS:
                if pack.has_program(W, name):
                    self.dh_progs[name] = self._load(name, "world")
            if not self.dh_progs:
                log.warn("--dh: the pack has no dh_terrain / dh_water program (Iris: 'incompatible pack')")
        # builtin fallback (Iris FallbackShader stand-in)
        fb = ProgInfo("<fallback>", "world")
        for ext, txt in (("vsh", FALLBACK_VSH), ("fsh", FALLBACK_FSH)):
            src = pack.Source("<fallback>", txt.split("\n"), [("<fallback>", i + 1) for i in range(txt.count("\n") + 1)])
            pack.build_compile_text(src, ext, "world", {})
            fb.stages[ext] = src
        fb.prog = glutil.Program("<fallback>", fb.stages, log, "world")
        fb.draw_buffers = [0]
        self.fallback = fb
        n = sum(1 for _ in self._all_infos())
        log.info(f"loaded {n} programs in {time.perf_counter() - t0:.1f} s, {len(self.failed)} failed"
                 + (": " + ", ".join(self.failed) if self.failed else ""))

    def _all_infos(self):
        for i in self.setup:
            yield i
        for i in self.shadow_computes:
            yield i
        for g in pack.COMPOSITE_GROUPS:
            for p in self.groups[g]:
                if p.frag:
                    yield p.frag
                for c in p.computes:
                    yield c
        if self.final.frag:
            yield self.final.frag
        for c in self.final.computes:
            yield c
        for i in self.world_progs.values():
            yield i
        for i in getattr(self, "dh_progs", {}).values():
            yield i

    # ================================================================== directives
    def _collect_directives(self):
        """ProgramSet.locateDirectives: fragment sources of shadowcomp, begin, prepare, gbuffers, deferred, composite."""
        log = self.log
        order = []
        for g in ("shadowcomp", "begin", "prepare"):
            order += [p.frag for p in self.groups[g] if p.frag]
        order += list(self.world_progs.values())
        for g in ("deferred", "composite"):
            order += [p.frag for p in self.groups[g] if p.frag]
        consts = {}
        for info in order:
            for d in info.consts:
                consts[d.key] = d
        self.consts = consts

        def cval(key, conv, default):
            d = consts.get(key)
            if d is None:
                return default
            v = conv(d.value)
            if v is None:
                log.warn(f"cannot parse const {d.type} {key} = {d.value}")
                return default
            return v

        def expand(v):
            # JCPP expands macros in directive values
            from .cpp import MiniCpp
            return MiniCpp(self.macros).expand(v).strip()

        self.target_fmt = {}
        self.target_clear = {}
        self.target_clear_color = {}
        for i in range(16):
            fmt, clear, cc = "RGBA", True, None
            for nm in [f"colortex{i}"] + ([LEGACY_TARGETS[i]] if i < 8 else []):
                d = consts.get(nm + "Format")
                if d is not None:
                    f = expand(d.value)
                    if glutil.format_info(f) is None:
                        log.warn(f"unknown format {f} for {nm}Format")
                    else:
                        fmt = f
                b = cval(nm + "Clear", pack.parse_bool, None)
                if b is not None:
                    clear = b
                c = cval(nm + "ClearColor", lambda v: pack.parse_vec(v, 4), None)
                if c is not None:
                    cc = c
            self.target_fmt[i] = fmt
            self.target_clear[i] = clear
            self.target_clear_color[i] = cc
        # shadow
        self.shadow_res = int(cval("shadowMapResolution", lambda v: int(float(v)), 1024))
        if self.cfg.shadow_res:
            if self.cfg.shadow_res != self.shadow_res:
                log.warn(f"--shadow-res {self.cfg.shadow_res} overrides shadowMapResolution {self.shadow_res} "
                         f"(shader constant left unchanged)")
            self.shadow_res = self.cfg.shadow_res
        self.shadow_distance = cval("shadowDistance", pack.parse_float, 160.0)
        self.shadow_near = cval("shadowNearPlane", pack.parse_float, -100.05)
        self.shadow_far = cval("shadowFarPlane", pack.parse_float, 156.0)
        self.shadow_interval = cval("shadowIntervalSize", pack.parse_float, 2.0)
        self.sun_path_rotation = cval("sunPathRotation", pack.parse_float, 0.0)
        self.voxel_distance = cval("voxelDistance", pack.parse_float, 0.0)
        self.half_lives = {k: cval(k, pack.parse_float, dv) for k, dv in
                           (("wetnessHalflife", 600.0), ("drynessHalflife", 200.0), ("eyeBrightnessHalflife", 10.0),
                            ("centerDepthHalflife", 1.0))}
        self.ambient_occlusion_level = cval("ambientOcclusionLevel", pack.parse_float, 1.0)
        hw = cval("shadowHardwareFiltering", pack.parse_bool, False)
        self.shadow_hw = [cval(f"shadowHardwareFiltering{i}", pack.parse_bool, hw) for i in range(2)]
        near_all = cval("shadowtexNearest", pack.parse_bool, False)
        self.shadow_nearest = [cval(f"shadowtex{i}Nearest", pack.parse_bool, near_all if i == 0 else False)
                               or cval(f"shadow{i}MinMagNearest", pack.parse_bool, False) for i in range(2)]
        self.shadowcolor_fmt, self.shadowcolor_clear, self.shadowcolor_cc, self.shadowcolor_nearest = {}, {}, {}, {}
        self.shadowcolor_mip = {}
        gen_mip = cval("generateShadowColorMipmap", pack.parse_bool, False)
        n_sc = 2
        for i in range(8):
            if any(k.startswith(f"shadowcolor{i}") for k in consts):
                n_sc = max(n_sc, i + 1)
        for info in self.world_progs.values():
            if info.name.startswith("shadow"):
                n_sc = max(n_sc, max(info.draw_buffers) + 1)
        self.n_shadowcolor = min(n_sc, 8)
        for i in range(self.n_shadowcolor):
            d = consts.get(f"shadowcolor{i}Format")
            self.shadowcolor_fmt[i] = expand(d.value) if d is not None and glutil.format_info(expand(d.value)) else "RGBA"
            self.shadowcolor_clear[i] = cval(f"shadowcolor{i}Clear", pack.parse_bool, True)
            self.shadowcolor_cc[i] = cval(f"shadowcolor{i}ClearColor", lambda v: pack.parse_vec(v, 4), [1.0] * 4)
            self.shadowcolor_nearest[i] = (cval(f"shadowcolor{i}Nearest", pack.parse_bool, False)
                                           or cval(f"shadowColor{i}Nearest", pack.parse_bool, False))
            self.shadowcolor_mip[i] = (cval(f"shadowcolor{i}Mipmap", pack.parse_bool, gen_mip)
                                       or cval(f"shadowColor{i}Mipmap", pack.parse_bool, False))
        log.section("directives")
        for i in range(16):
            cc = self.target_clear_color[i]
            log.info(f"  colortex{i:<2d} {self.target_fmt[i]:10s} clear={str(self.target_clear[i]):5s} "
                     f"clearColor={'default' if cc is None else tuple(cc)}", echo=self.cfg.verbose)
        log.info(f"  shadow: res {self.shadow_res} distance {self.shadow_distance} interval {self.shadow_interval} "
                 f"near/far {self.shadow_near}/{self.shadow_far} sunPathRotation {self.sun_path_rotation} "
                 f"voxelDistance {self.voxel_distance} hwFiltering {self.shadow_hw}")
        for i in range(self.n_shadowcolor):
            log.info(f"  shadowcolor{i} {self.shadowcolor_fmt[i]} clear={self.shadowcolor_clear[i]} "
                     f"clearColor={tuple(self.shadowcolor_cc[i])}", echo=self.cfg.verbose)

    # ================================================================== schedule (flip semantics)
    def _build_schedule(self):
        log = self.log
        flipper = set()

        def toggle(b):
            if b in flipper:
                flipper.remove(b)
            else:
                flipper.add(b)

        snapshots = {}
        for g in ("begin", "prepare", "deferred", "composite"):
            for b, f in self.props.flips.get(g + "_pre", {}).items():
                if f:
                    toggle(b)
            for p in self.groups[g]:
                p.reads_alt = frozenset(flipper)
                if p.frag is None:
                    continue
                db = p.frag.draw_buffers
                p.writes = [(b, "main" if b in p.reads_alt else "alt") for b in db]
                explicit = self.props.flips.get(p.name, {})
                for b in db:
                    if explicit.get(b) is False:
                        continue
                    toggle(b)
                for b, f in explicit.items():
                    if f:
                        toggle(b)
                p.scale = self.props.scale.get(p.name, (1.0, 0.0, 0.0))
                for d in p.frag.consts:
                    if d.type == "bool" and d.key.endswith("MipmapEnabled"):
                        nm = d.key[:-len("MipmapEnabled")]
                        idx = int(nm[8:]) if nm.startswith("colortex") and nm[8:].isdigit() else (
                            LEGACY_TARGETS.index(nm) if nm in LEGACY_TARGETS else None)
                        if idx is not None:
                            if pack.parse_bool(d.value):
                                p.mip.add(idx)
                            else:
                                p.mip.discard(idx)
                p.blend = self.props.blend_program.get(p.name, "default")
            snapshots[g] = frozenset(flipper)
        self.flipped_before_shadow = snapshots["begin"]
        self.flipped_after_prepare = snapshots["prepare"]
        self.flipped_after_translucent = snapshots["deferred"]
        self.final.reads_alt = snapshots["composite"]
        if self.final.frag:
            for d in self.final.frag.consts:
                if d.type == "bool" and d.key.startswith("colortex") and d.key.endswith("MipmapEnabled"):
                    if pack.parse_bool(d.value):
                        self.final.mip.add(int(d.key[8:-len("MipmapEnabled")]))
        # shadowcomp flips over shadowcolor buffers
        sflip = set()
        for p in self.groups["shadowcomp"]:
            p.reads_alt = frozenset(sflip)
            if p.frag:
                p.writes = [(b, "main" if b in p.reads_alt else "alt") for b in p.frag.draw_buffers]
                for b in p.frag.draw_buffers:
                    if b in sflip:
                        sflip.remove(b)
                    else:
                        sflip.add(b)
                p.scale = self.props.scale.get(p.name, (1.0, 0.0, 0.0))
        self.shadow_flipped_final = frozenset(sflip)
        # final swap list: buffers read from alt at the end that are not cleared each frame
        self.swap = sorted(b for b in self.final.reads_alt if not self.target_clear.get(b, True))
        log.section("schedule")
        for g in pack.COMPOSITE_GROUPS:
            for p in self.groups[g]:
                desc = []
                if p.computes:
                    desc.append("compute[" + ", ".join(c.name for c in p.computes) + "]")
                if p.frag:
                    desc.append(f"RT {p.frag.draw_buffers} -> " + ",".join(f"{b}{'M' if w == 'main' else 'A'}" for b, w in p.writes))
                    if p.scale[0] != 1.0:
                        desc.append(f"scale {p.scale[0]}")
                    if p.mip:
                        desc.append(f"mip {sorted(p.mip)}")
                desc.append(f"readsAlt {sorted(p.reads_alt)}")
                log.info(f"  {p.name:14s} " + "  ".join(desc), echo=self.cfg.verbose)
        log.info(f"  final: readsAlt {sorted(self.final.reads_alt)}  swap(alt->main) {self.swap}", echo=self.cfg.verbose)

    # ================================================================== resources
    def create_resources(self, atlas, lightmap_img, meshes):
        log = self.log
        W, H = self.w, self.h
        self.targets = []
        for i in range(16):
            tw, th = W, H
            sb = self.props.size_buffer.get(i)
            if sb:
                def dim(spec, full):
                    return int(float(spec) * full) if "." in spec else int(spec)
                tw, th = dim(sb[0], W), dim(sb[1], H)
            self.targets.append(RenderTarget(i, self.target_fmt[i], tw, th))
        self.depth = {k: glutil.Texture(GL.GL_TEXTURE_2D, "DEPTH_COMPONENT32F", W, H, label=k, depth=True)
                      for k in ("depthtex0", "depthtex1", "depthtex2")}
        if self.dh:
            # Distant Horizons' own depth buffer (dhDepthTex0) and its copy before translucent LODs (dhDepthTex1)
            for k in ("dhDepthTex0", "dhDepthTex1"):
                self.depth[k] = glutil.Texture(GL.GL_TEXTURE_2D, "DEPTH_COMPONENT32F", W, H, label=k, depth=True)
        R = self.shadow_res
        self.shadowtex = [glutil.Texture(GL.GL_TEXTURE_2D, "DEPTH_COMPONENT32F", R, R, label=f"shadowtex{i}", depth=True)
                          for i in range(2)]
        self.shadowcolor = []
        for i in range(self.n_shadowcolor):
            fmt = self.shadowcolor_fmt[i]
            self.shadowcolor.append((glutil.Texture(GL.GL_TEXTURE_2D, fmt, R, R, label=f"shadowcolor{i}"),
                                     glutil.Texture(GL.GL_TEXTURE_2D, fmt, R, R, label=f"shadowcolor{i}_alt")))
        # custom images
        self.images = {}
        self.image_samplers = {}
        for img in self.props.images:
            try:
                if img["relative"]:
                    size = [max(1, int(W * img["rel"][0])), max(1, int(H * img["rel"][1]))]
                else:
                    size = img["size"]
                target = {1: GL.GL_TEXTURE_1D, 2: GL.GL_TEXTURE_2D, 3: GL.GL_TEXTURE_3D}[len(size)]
                t = glutil.Texture(target, img["internal"], *size, label=img["name"])
                t.clear((0, 0, 0, 0))
                self.images[img["name"]] = (t, img)
                if img["sampler"]:
                    self.image_samplers[img["sampler"]] = t
                log.info(f"  image {img['name']:20s} {img['internal']:8s} {'x'.join(map(str, size)):12s} "
                         f"clear={img['clear']} sampler={img['sampler']}", echo=self.cfg.verbose)
            except Exception as e:  # noqa: BLE001
                log.error(f"image.{img['name']}: {e}")
        # SSBOs
        self.ssbos = {}
        for idx, (size, rel) in sorted(self.props.buffer_objects.items()):
            if rel is not None and rel[0]:
                # relative SSBO: size per pixel of a (W*sx) x (H*sy) screen
                size = int(size * math.ceil(W * rel[1]) * math.ceil(H * rel[2]))
            b = int(GL.glGenBuffers(1))
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, b)
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, size, np.zeros(size, np.uint8), GL.GL_DYNAMIC_COPY)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, idx, b)
            self.ssbos[idx] = (b, size)
            log.info(f"  bufferObject.{idx} = {size} bytes", echo=self.cfg.verbose)
        # custom textures
        self.custom_textures = {}
        for sname, spec in self.props.custom_textures.items():
            try:
                r = self._load_custom_texture(sname, spec)
            except Exception as e:  # noqa: BLE001 — a bad texture line must not abort the run
                log.error(f"customTexture.{sname} = {spec}: {type(e).__name__}: {e}")
                r = None
            if r is not None:
                self.custom_textures[sname] = r
        # noise texture (Iris default: 256^2 random)
        rng = np.random.default_rng(12345)
        nz = rng.integers(0, 256, (256, 256, 4), dtype=np.uint8)
        nz[..., 3] = 255
        self.noisetex = glutil.Texture(GL.GL_TEXTURE_2D, "RGBA8", 256, 256, label="noisetex", data=nz,
                                       data_fmt=GL.GL_RGBA, data_type=GL.GL_UNSIGNED_BYTE)
        # world textures
        def tex_from(img01, label, mips=True):
            u8 = np.ascontiguousarray(np.clip(img01 * 255 + 0.5, 0, 255).astype(np.uint8))
            t = glutil.Texture(GL.GL_TEXTURE_2D, "RGBA8", u8.shape[1], u8.shape[0], label=label, data=u8,
                               data_fmt=GL.GL_RGBA, data_type=GL.GL_UNSIGNED_BYTE)
            if mips:
                GL.glTextureParameteri(t.id, GL.GL_TEXTURE_MAX_LEVEL, 4)
                t.generate_mipmaps()
            return t
        self.gtexture = tex_from(atlas.albedo, "gtexture")
        self.normals_tex = tex_from(atlas.normal, "normals")
        self.specular_tex = tex_from(atlas.specular, "specular")
        self.lightmap_tex = tex_from(lightmap_img, "lightmap", mips=False)
        self.atlas_size = (atlas.size, atlas.size)
        # dummies
        self.dummies = {}
        # output (Minecraft main framebuffer stand-in)
        self.output = glutil.Texture(GL.GL_TEXTURE_2D, "RGBA8", W, H, label="output")
        # meshes
        self.meshes = {k: Mesh(v, sortable=(k in ("translucent", "dh_water"))) for k, v in meshes.items()}
        # fullscreen quad
        q = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 1, 0, 1, 1], [0, 1, 0, 0, 1]], np.float32)
        self.quad_vao = int(GL.glGenVertexArrays(1))
        self.quad_vbo = int(GL.glGenBuffers(1))
        self.quad_ibo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(self.quad_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.quad_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, q.nbytes, q, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 20, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, 20, ctypes.c_void_p(12))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.quad_ibo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, 24, np.array([0, 1, 2, 2, 3, 0], np.uint32), GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)
        # composite framebuffers
        for g in pack.COMPOSITE_GROUPS:
            for p in self.groups[g]:
                if p.frag:
                    if g == "shadowcomp":
                        texs = [self.shadowcolor[b][0 if w == "main" else 1] for b, w in p.writes if b < len(self.shadowcolor)]
                    else:
                        texs = [self.targets[b].main if w == "main" else self.targets[b].alt for b, w in p.writes]
                    p.fbo = self.fbo(texs, None)
        self.final_fbo = self.fbo([self.output], None)
        self.first_frame = True
        log.info(f"resources: {W}x{H}, shadow {R}^2, {len(self.images)} images, {len(self.ssbos)} SSBOs, "
                 f"{len(self.custom_textures)} custom textures")

    # Iris raw custom textures (ShaderProperties customTexture.*, ShaderPack.readTexture, GlTexture)
    RAW_TEXTURE_TYPES = {"TEXTURE_1D": (GL.GL_TEXTURE_1D, 1), "TEXTURE_2D": (GL.GL_TEXTURE_2D, 2),
                         "TEXTURE_RECTANGLE": (GL.GL_TEXTURE_RECTANGLE, 2), "TEXTURE_3D": (GL.GL_TEXTURE_3D, 3)}
    # Iris PixelFormat: component count
    RAW_PIXEL_FORMATS = {"RED": 1, "RG": 2, "RGB": 3, "BGR": 3, "RGBA": 4, "BGRA": 4, "RED_INTEGER": 1,
                         "RG_INTEGER": 2, "RGB_INTEGER": 3, "BGR_INTEGER": 3, "RGBA_INTEGER": 4, "BGRA_INTEGER": 4}
    # Iris PixelType: bytes per component, or per pixel for the packed types (names with a bit layout)
    RAW_PIXEL_TYPES = {"BYTE": 1, "SHORT": 2, "INT": 4, "HALF_FLOAT": 2, "FLOAT": 4, "UNSIGNED_BYTE": 1,
                       "UNSIGNED_SHORT": 2, "UNSIGNED_INT": 4, "UNSIGNED_BYTE_3_3_2": 1, "UNSIGNED_BYTE_2_3_3_REV": 1,
                       "UNSIGNED_SHORT_5_6_5": 2, "UNSIGNED_SHORT_5_6_5_REV": 2, "UNSIGNED_SHORT_4_4_4_4": 2,
                       "UNSIGNED_SHORT_4_4_4_4_REV": 2, "UNSIGNED_SHORT_5_5_5_1": 2, "UNSIGNED_SHORT_1_5_5_5_REV": 2,
                       "UNSIGNED_INT_8_8_8_8": 4, "UNSIGNED_INT_8_8_8_8_REV": 4, "UNSIGNED_INT_10_10_10_2": 4,
                       "UNSIGNED_INT_2_10_10_10_REV": 4, "UNSIGNED_INT_10F_11F_11F_REV": 4,
                       "UNSIGNED_INT_5_9_9_9_REV": 4}

    def _load_custom_texture(self, sname, spec):
        """One `customTexture.<sampler> = ...` line -> (Texture, sampler object), or None (logged).

        PNG: `path` (RGBA8 as stored: not sRGB-decoded, not premultiplied; PNG row 0 at t = 0).
        Raw (Iris 1.7+): `path TEXTURE_1D|TEXTURE_2D|TEXTURE_RECTANGLE|TEXTURE_3D internal w [h [d]] format type`;
        Iris picks the dimensions by the field count (6: w, 7: w h, 8: w h d). The file holds the texels
        tightly packed, x fastest, then y, then z; its size must be w*h*d * bytes per pixel. Uploaded with
        GL_UNPACK_ALIGNMENT 1 (Iris uses 4: identical whenever a row is a multiple of 4 bytes, warned
        otherwise).
        Filtering (`<path>.mcmeta`: {"texture": {"blur": b, "clamp": c}}): defaults blur = clamp = false
        for PNGs and blur = clamp = true for raw textures (ShaderPack.readTexture). Sampled through a sampler
        object like Iris' GlSampler: blur -> LINEAR else NEAREST (min and mag, no mipmaps), clamp -> S and T
        CLAMP_TO_EDGE else REPEAT, and R always REPEAT (GlSampler never sets WRAP_R)."""
        log = self.log
        parts = spec.split()
        rel = parts[0][1:] if parts[0].startswith("/") else parts[0]   # Iris strips one leading slash
        path = os.path.join(pack.SHADERS, rel)
        raw = len(parts) > 1
        if raw and len(parts) not in (6, 7, 8):
            log.warn(f"customTexture.{sname} = {spec}: unknown texture directive (Iris: 6, 7 or 8 fields), ignored")
            return None
        if not os.path.isfile(path):
            log.warn(f"customTexture.{sname}: {rel} not found")
            return None
        blur = clamp = raw
        meta = path + ".mcmeta"
        meta_note = "no .mcmeta"
        if os.path.isfile(meta):
            import json
            try:
                with open(meta, encoding="utf-8") as f:
                    m = json.load(f).get("texture") or {}
                if "blur" in m:
                    blur = bool(m["blur"])
                if "clamp" in m:
                    clamp = bool(m["clamp"])
                meta_note = os.path.basename(meta)
            except Exception as e:  # noqa: BLE001 — Iris logs and keeps the defaults
                log.warn(f"customTexture.{sname}: unreadable {os.path.basename(meta)} ({e}); defaults kept")
        filt = "linear" if blur else "nearest"
        wrap = "clamp" if clamp else "repeat"
        smp = self.samplers.get(filt, filt, wrap, wrap_r="repeat")
        how = (f"blur={blur} clamp={clamp} ({meta_note}) -> {'LINEAR' if blur else 'NEAREST'}, "
               f"S/T {'CLAMP_TO_EDGE' if clamp else 'REPEAT'}, R REPEAT")
        if not raw:
            im = np.asarray(Image.open(path).convert("RGBA"), np.uint8)
            t = glutil.Texture(GL.GL_TEXTURE_2D, "RGBA8", im.shape[1], im.shape[0], label=sname,
                               data=np.ascontiguousarray(im), data_fmt=GL.GL_RGBA, data_type=GL.GL_UNSIGNED_BYTE)
            log.info(f"  customTexture.{sname} = {rel} PNG {im.shape[1]}x{im.shape[0]} {how}", echo=self.cfg.verbose)
            return t, smp
        ttype, internal, fmt, ptype = parts[1].upper(), parts[2], parts[-2].upper(), parts[-1].upper()
        sizes = [int(x) for x in parts[3:-2]]
        if ttype not in self.RAW_TEXTURE_TYPES:
            log.error(f"customTexture.{sname}: unknown texture type {parts[1]}")
            return None
        target, dims = self.RAW_TEXTURE_TYPES[ttype]
        if dims != len(sizes):
            log.error(f"customTexture.{sname}: {ttype} with {len(sizes)} size field(s); Iris takes the sizes "
                      f"from the field count and would create a degenerate texture")
            return None
        finfo = glutil.format_info(internal)
        if finfo is None:
            log.error(f"customTexture.{sname}: unknown internal format {internal}")
            return None
        if fmt not in self.RAW_PIXEL_FORMATS or ptype not in self.RAW_PIXEL_TYPES:
            log.error(f"customTexture.{sname}: unknown pixel format / type {parts[-2]} {parts[-1]}")
            return None
        packed = any(ch.isdigit() for ch in ptype)
        bpp = self.RAW_PIXEL_TYPES[ptype] * (1 if packed else self.RAW_PIXEL_FORMATS[fmt])
        need = bpp * int(np.prod(sizes))
        data = open(path, "rb").read()
        if len(data) < need:
            log.error(f"customTexture.{sname}: {rel} has {len(data)} bytes, {ttype} "
                      f"{'x'.join(map(str, sizes))} {fmt} {ptype} needs {need}; not loaded")
            return None
        if len(data) > need:
            log.warn(f"customTexture.{sname}: {rel} has {len(data)} bytes, only the first {need} are used")
            data = data[:need]
        row = sizes[0] * bpp
        if dims > 1 and row % 4:
            log.warn(f"customTexture.{sname}: rows of {row} bytes; Iris uploads raw textures with "
                     f"GL_UNPACK_ALIGNMENT 4 (rows padded to {row + 4 - row % 4} bytes), irisim tightly packed")
        if finfo["kind"] in ("int", "uint") and blur:
            log.warn(f"customTexture.{sname}: integer format {internal} with blur = true: LINEAR filtering "
                     f"makes the texture incomplete (reads return 0), as in Iris")
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        try:
            t = glutil.Texture(target, internal, *sizes, label=sname, data=bytes(data),
                               data_fmt=getattr(GL, "GL_" + fmt), data_type=getattr(GL, "GL_" + ptype))
        finally:
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        log.info(f"  customTexture.{sname} = {rel} {ttype} {finfo['name']} {'x'.join(map(str, sizes))} {fmt} "
                 f"{ptype} ({len(data)} B) {how}", echo=self.cfg.verbose)
        return t, smp

    def fbo(self, color_texs, depth_tex):
        key = (tuple(t.id for t in color_texs), depth_tex.id if depth_tex else 0)
        if key in self.fbo_cache:
            return self.fbo_cache[key]
        f = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, f)
        for i, t in enumerate(color_texs):
            GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0 + i, t.id, 0)
        if depth_tex is not None:
            GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, depth_tex.id, 0)
        if color_texs:
            GL.glDrawBuffers(len(color_texs), np.array([GL.GL_COLOR_ATTACHMENT0 + i for i in range(len(color_texs))], np.uint32))
        else:
            GL.glDrawBuffer(GL.GL_NONE)
        st = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if st != GL.GL_FRAMEBUFFER_COMPLETE:
            self.log.error(f"framebuffer incomplete ({st}) for {[t.label for t in color_texs]}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self.fbo_cache[key] = f
        return f

    # ================================================================== binding
    def dummy(self, target, kind, shadow=False):
        key = (target, kind, shadow)
        if key not in self.dummies:
            if shadow:
                t = glutil.Texture(GL.GL_TEXTURE_2D, "DEPTH_COMPONENT32F", 1, 1, label="dummy_depth", depth=True)
                t.clear((1.0, 0, 0, 0))
            else:
                fmt = {"float": "RGBA8", "uint": "R32UI", "int": "R32I"}[kind]
                tgt = target if target in (GL.GL_TEXTURE_2D, GL.GL_TEXTURE_3D, GL.GL_TEXTURE_1D) else GL.GL_TEXTURE_2D
                t = glutil.Texture(tgt, fmt, 1, 1, 1, label="dummy")
                t.clear((0, 0, 0, 0))
            self.dummies[key] = t
        return self.dummies[key]

    def _colortex_index(self, name):
        if name.startswith("colortex") and name[8:].isdigit():
            i = int(name[8:])
            return i if i < 16 else None
        if name in LEGACY_TARGETS:
            return LEGACY_TARGETS.index(name)
        return None

    def target_sampler(self, tex):
        if tex.is_integer:
            return self.samplers.get("nearest", "nearest")
        if tex.id in self.mip_enabled:
            return self.samplers.get("linear_mip", "linear")
        return self.samplers.get("linear", "linear")

    def resolve_sampler(self, name, tname, ctx, prog):
        target, kind, is_shadow_type = glutil.sampler_target(tname)
        S = self.samplers
        i = self._colortex_index(name)
        if i is not None:
            t = self.targets[i].alt if i in ctx["reads_alt"] else self.targets[i].main
            return t, self.target_sampler(t)
        if name in ("depthtex0", "gdepthtex"):
            return self.depth["depthtex0"], S.get("nearest", "nearest")
        if name in ("depthtex1", "depthtex2"):
            return self.depth[name], S.get("nearest", "nearest")
        if name in ("dhDepthTex", "dhDepthTex0", "dhDepthTex1") and "dhDepthTex0" in self.depth:
            return self.depth["dhDepthTex1" if name == "dhDepthTex1" else "dhDepthTex0"], S.get("nearest", "nearest")
        if name in ("shadowtex0", "shadowtex1", "shadow", "watershadow"):
            if name == "shadow":
                idx = 1 if any(s[0] == "watershadow" for s in prog.samplers) else 0
            elif name == "watershadow":
                idx = 0
            else:
                idx = int(name[-1])
            t = self.shadowtex[idx]
            if is_shadow_type:
                f = "linear" if self.shadow_hw[idx] else "nearest"
                return t, S.get(f, f, "clamp", compare=True)
            f = "nearest" if self.shadow_nearest[idx] else "linear"
            return t, S.get(f, f)
        if name.startswith("shadowcolor") and (name == "shadowcolor" or name[11:].isdigit()):
            idx = 0 if name == "shadowcolor" else int(name[11:])
            if idx < len(self.shadowcolor):
                t = self.shadowcolor[idx][1 if idx in ctx.get("shadow_reads_alt", frozenset()) else 0]
                if t.is_integer or self.shadowcolor_nearest.get(idx):
                    return t, S.get("nearest", "nearest")
                if self.shadowcolor_mip.get(idx) and t.id in self.mip_enabled:
                    return t, S.get("linear_mip", "linear")
                return t, S.get("linear", "linear")
            return None
        if name == "noisetex":
            return self.noisetex, S.get("linear", "linear", "repeat")
        if name in ("gtexture", "texture", "tex"):
            return ctx.get("gtexture", self.gtexture), S.get("nearest_mip_linear", "nearest", "repeat")
        if name == "normals":
            return self.normals_tex, S.get("nearest_mip_linear", "nearest", "repeat")
        if name == "specular":
            return self.specular_tex, S.get("nearest_mip_linear", "nearest", "repeat")
        if name == "lightmap":
            return self.lightmap_tex, S.get("linear", "linear")
        if name in self.custom_textures:
            t, smp = self.custom_textures[name]
            if t.target != target:
                dims = {GL.GL_TEXTURE_1D: "1D", GL.GL_TEXTURE_2D: "2D", GL.GL_TEXTURE_3D: "3D",
                        GL.GL_TEXTURE_RECTANGLE: "RECTANGLE"}.get(t.target, str(t.target))
                self.log.once(("ctype", name, prog.name), f"ERROR custom texture {name}: {prog.name} declares "
                              f"{tname}, the texture is {dims} (bound to a black dummy)")
                return None
            return t, smp
        if name in self.image_samplers:
            t = self.image_samplers[name]
            return t, S.get("nearest", "nearest") if t.is_integer else S.get("linear", "linear")
        return None

    def resolve_image(self, name, ctx):
        if name in self.images:
            t, _ = self.images[name]
            return t
        if name.startswith("colorimg") and name[8:].isdigit():
            i = int(name[8:])
            if i < 16:
                return self.targets[i].alt if i in ctx["reads_alt"] else self.targets[i].main
        if name.startswith("shadowcolorimg") and name[14:].isdigit():
            i = int(name[14:])
            if i < len(self.shadowcolor):
                return self.shadowcolor[i][1 if i in ctx.get("shadow_reads_alt", frozenset()) else 0]
        return None

    def bind(self, info, ctx, uniforms):
        prog = info.prog
        prog.use()
        for name, (loc, typ) in prog.uniforms.items():
            if name in uniforms:
                if not glutil.set_uniform(loc, typ, uniforms[name]):
                    self.log.once(("utype", info.name, name), f"  uniform {name} in {info.name}: unsupported type {typ}")
            else:
                self.missing_uniforms.setdefault(name, set()).add(info.name)
        for (name, loc, tname, unit) in prog.samplers:
            r = self.resolve_sampler(name, tname, ctx, prog)
            if r is None:
                self.missing_samplers.setdefault(name, set()).add(info.name)
                target, kind, sh = glutil.sampler_target(tname)
                t = self.dummy(target, kind, sh)
                smp = self.samplers.get("nearest", "nearest", "clamp", compare=sh)
            else:
                t, smp = r
            GL.glBindTextureUnit(unit, t.id)
            GL.glBindSampler(unit, smp)
        for (name, loc, tname, unit) in prog.images:
            t = self.resolve_image(name, ctx)
            if t is None:
                self.missing_samplers.setdefault(name, set()).add(info.name)
                kind = "uint" if tname.startswith("UNSIGNED") else ("int" if tname.startswith("INT_") else "float")
                tgt = GL.GL_TEXTURE_3D if "3D" in tname else GL.GL_TEXTURE_2D
                t = self.dummy(tgt, kind)
            layered = GL.GL_TRUE if t.target == GL.GL_TEXTURE_3D else GL.GL_FALSE
            if t.info["name"] not in glutil.IMAGE_FORMATS_OK:
                self.log.once(("imgfmt", name), f"WARN  image {name}: format {t.info['name']} not usable as image")
                continue
            GL.glBindImageTexture(unit, t.id, 0, layered, 0, GL.GL_READ_WRITE, t.info["internal"])

    # ================================================================== state helpers
    def apply_blend(self, default, override, buffer_overrides, draw_buffers):
        mode = default
        if override != "default":
            mode = None if override is None else tuple(BLEND_FUNCS[m] for m in override)
        if mode is None:
            GL.glDisable(GL.GL_BLEND)
        else:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFuncSeparate(*mode)
        for buf, m in (buffer_overrides or {}).items():
            if buf in draw_buffers:
                slot = draw_buffers.index(buf)
                if m is None:
                    GL.glDisablei(GL.GL_BLEND, slot)
                else:
                    GL.glEnablei(GL.GL_BLEND, slot)
                    GL.glBlendFuncSeparatei(slot, *[BLEND_FUNCS[x] for x in m])

    def timed(self, name):
        pipe = self

        class _T:
            def __enter__(self_):
                if pipe.cfg.timing:
                    GL.glFinish()
                self_.t = time.perf_counter()

            def __exit__(self_, *a):
                if pipe.cfg.timing:
                    GL.glFinish()
                dt = (time.perf_counter() - self_.t) * 1000
                pipe.timings.setdefault(name, []).append(dt)
        return _T()

    def barrier(self):
        GL.glMemoryBarrier(GL.GL_ALL_BARRIER_BITS)

    def safe(self, what, fn, *a, **kw):
        """Run one pass/draw; a GL error or exception is logged and the frame continues."""
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            import traceback
            msg = f"{what}: {type(e).__name__}: {e}"
            self.log.error(msg[:600])
            self.log.once(("tb", what), traceback.format_exc())
            try:
                while GL.glGetError() != GL.GL_NO_ERROR:
                    pass
            except Exception:  # noqa: BLE001
                pass
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            GL.glUseProgram(0)
            GL.glDisable(GL.GL_BLEND)
            return None

    # ================================================================== passes
    def dispatch(self, info, ctx, uniforms, w, h):
        if not info.ok:
            return
        lx, ly, lz = info.prog.local_size
        if info.work_groups is not None:
            g = info.work_groups
        elif info.work_groups_rel is not None:
            g = (int(math.ceil(math.ceil(w * info.work_groups_rel[0]) / lx)),
                 int(math.ceil(math.ceil(h * info.work_groups_rel[1]) / ly)), 1)
        else:
            g = (int(math.ceil(w / lx)), int(math.ceil(h / ly)), 1)
        self.barrier()
        self.bind(info, ctx, uniforms)
        GL.glDispatchCompute(max(1, g[0]), max(1, g[1]), max(1, g[2]))
        self.barrier()

    def draw_quad(self):
        GL.glBindVertexArray(self.quad_vao)
        GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

    def run_group(self, g, uniforms):
        for p in self.groups[g]:
            ctx = {"reads_alt": p.reads_alt, "stage": g, "shadow_reads_alt": self.shadow_flipped_final}
            if g == "shadowcomp":
                ctx = {"reads_alt": self.flipped_before_shadow, "shadow_reads_alt": p.reads_alt, "stage": g}
            # timings per program: each compute (prepare1, prepare1_a, ...) under its own name, the fragment
            # program under the pass name (a base compute of the same name then reports as <name>.csh)
            for c in p.computes:
                key = c.name + ".csh" if (c.name == p.name and p.frag is not None) else c.name
                with self.timed(key):
                    if g == "shadowcomp":
                        self.safe(c.name, self.dispatch, c, ctx, uniforms, self.shadow_res, self.shadow_res)
                    else:
                        self.safe(c.name, self.dispatch, c, ctx, uniforms, self.w, self.h)
            if p.frag is not None:
                with self.timed(p.name):
                    self.safe(p.name, self._fullscreen, p, ctx, uniforms, g == "shadowcomp")
            if self.frame_dump:
                self._dump_pass(p, g)
                self._dump_bound_images(p)
            self._npy_seq_pass(p, g)

    def _fullscreen(self, p, ctx, uniforms, shadow=False):
        if not shadow:
            for b in p.mip:
                t = self.targets[b].alt if b in p.reads_alt else self.targets[b].main
                t.generate_mipmaps()
                self.mip_enabled.add(t.id)
        if shadow:
            vw = vh = self.shadow_res
        else:
            t0 = self.targets[p.writes[0][0]] if p.writes else self.targets[0]
            vw, vh = t0.w, t0.h
        s, ox, oy = p.scale
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, p.fbo)
        GL.glViewport(int(vw * ox), int(vh * oy), int(vw * s), int(vh * s))
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glColorMask(True, True, True, True)
        if not p.frag.ok:
            for i in range(len(p.writes)):
                GL.glClearBufferfv(GL.GL_COLOR, i, np.zeros(4, np.float32))
            return
        self.apply_blend(None, p.blend, None, p.frag.draw_buffers)
        self.bind(p.frag, ctx, uniforms)
        self.draw_quad()
        GL.glDisable(GL.GL_BLEND)

    def _npy_seq_on(self):
        """--npy-seq is active for the current frame."""
        return bool(getattr(self.cfg, "npy_seq", None)) and self.frame_index >= getattr(self.cfg, "npy_seq_from", 0)

    def _npy_seq_pass(self, p, g):
        """--npy-seq: right after pass p, save each colortex it wrote whose label <pass>_colortex<N> is listed,
        as float32 RGBA (GL row order, NaN/Inf zeroed, clamped to +-65000) per frame."""
        if g == "shadowcomp" or not self._npy_seq_on():
            return
        for b, w in p.writes:
            lab = f"{p.name}_colortex{b}"
            if lab in self.cfg.npy_seq:
                t = self.targets[b].main if w == "main" else self.targets[b].alt
                a = t.read_rgba()
                os.makedirs(self.cfg.npy_seq_dir, exist_ok=True)
                np.save(os.path.join(self.cfg.npy_seq_dir, f"{lab}_f{self.frame_index:03d}.npy"),
                        np.clip(np.nan_to_num(a), -65000, 65000).astype(np.float32))

    def _dump_pass(self, p, g):
        d = self.dumper
        for b, w in p.writes:
            if g == "shadowcomp":
                if b < len(self.shadowcolor):
                    d.texture(f"{p.name}_shadowcolor{b}", self.shadowcolor[b][0 if w == "main" else 1])
            else:
                t = self.targets[b].main if w == "main" else self.targets[b].alt
                d.texture(f"{p.name}_colortex{b}", t)

    # ------------------------------------------------------------------ world geometry
    def resolve_world(self, name):
        n = name
        while n is not None:
            if n in self.world_progs:
                return self.world_progs[n]
            if n in self.dh_progs:
                return self.dh_progs[n]
            n = pack.GBUFFER_FALLBACK.get(n)
        return self.fallback

    def draw_world(self, prog_name, mesh, *a, **kw):
        r = self.safe(f"draw {prog_name}", self._draw_world, prog_name, mesh, *a, **kw)
        return r if r is not None else self.resolve_world(prog_name)

    def _draw_world(self, prog_name, mesh, uniforms, per_draw, *, shadow=False, translucent=False, depth_mask=True,
                    depth_test=True, cull=True, default_blend=None, depth_tex=None):
        info = self.resolve_world(prog_name)
        if mesh is None or mesh.count == 0:
            return info
        if not info.ok:
            self.log.once(("skipdraw", info.name), f"WARN  {info.name} failed to compile — draws skipped")
            return info
        db = info.draw_buffers
        if shadow:
            texs = [self.shadowcolor[b][0] for b in db if b < len(self.shadowcolor)]
            fbo = self.fbo(texs, self.shadowtex[0])
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glViewport(0, 0, self.shadow_res, self.shadow_res)
            reads_alt = self.flipped_before_shadow
        else:
            reads_alt = self.flipped_after_translucent if translucent else self.flipped_after_prepare
            texs = [self.targets[b].alt if b in reads_alt else self.targets[b].main for b in db]
            fbo = self.fbo(texs, depth_tex if depth_tex is not None else self.depth["depthtex0"])
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glViewport(0, 0, self.w, self.h)
        if depth_test:
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE if depth_mask else GL.GL_FALSE)
        if cull:
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glCullFace(GL.GL_BACK)
            GL.glFrontFace(GL.GL_CCW)
        else:
            GL.glDisable(GL.GL_CULL_FACE)
        if shadow:
            # Iris: blending disabled in the shadow pass (BlendModeOverride.OFF default for shadow programs)
            self.apply_blend(None, self.props.blend_program.get(info.name, None), self.props.blend_buffer.get(info.name), db)
        else:
            self.apply_blend(default_blend, self.props.blend_program.get(info.name, "default"),
                             self.props.blend_buffer.get(info.name), db)
        u = dict(uniforms)
        u.update(per_draw)
        ctx = {"reads_alt": reads_alt, "stage": "shadow" if shadow else "gbuffers",
               "shadow_reads_alt": self.shadow_flipped_final}
        self.bind(info, ctx, u)
        mesh.draw()
        GL.glDisable(GL.GL_BLEND)
        return info

    # ================================================================== frame
    def clear_targets(self, fog):
        full = self.first_frame
        for i, t in enumerate(self.targets):
            if not (full or self.target_clear[i]):
                continue
            cc = self.target_clear_color[i]
            if cc is None:
                cc = (fog[0], fog[1], fog[2], 1.0) if i == 0 else ((1, 1, 1, 1) if i == 1 else (0, 0, 0, 0))
            t.main.clear(cc)
            t.alt.clear(cc)

    def render_frame(self, fs, dump=False):
        """fs: FrameState from cli.py (uniforms + per-draw matrices + scene)."""
        self.frame_dump = dump
        U = fs.uniforms
        self.mip_enabled.clear()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        # ---- setup (once, after a full clear)
        if self.first_frame:
            with self.timed("clear(full)"):
                self.clear_targets(U["fogColor"])
                for i in range(len(self.shadowcolor)):
                    for t in self.shadowcolor[i]:
                        t.clear(self.shadowcolor_cc[i])
            for info in self.setup:
                with self.timed(info.name):
                    self.safe(info.name, self.dispatch, info, {"reads_alt": frozenset(), "stage": "setup"}, U, 1, 1)
                if dump and info.ok:
                    self._dump_images(f"after_{info.name}", only={n for n, *_ in info.prog.images})
        # ---- beginLevelRendering
        with self.timed("clear"):
            for name, (t, img) in self.images.items():
                if img["clear"]:
                    t.clear((0, 0, 0, 0))
            self.shadowtex[0].clear((1.0, 0, 0, 0))
            for c in self.shadow_computes:
                self.safe(c.name, self.dispatch, c, {"reads_alt": self.flipped_before_shadow, "stage": "shadow"}, U,
                          self.shadow_res, self.shadow_res)
            for i in range(len(self.shadowcolor)):
                if self.first_frame or self.shadowcolor_clear[i]:
                    for t in self.shadowcolor[i]:
                        t.clear(self.shadowcolor_cc[i])
            if not self.first_frame:
                self.clear_targets(U["fogColor"])
        self.run_group("begin", U)
        prepare_first = self.props.bool_flag("prepareBeforeShadow", False)
        if prepare_first:
            self.run_group("prepare", U)
        # ---- shadow pass
        has_shadow_prog = any(k.startswith("shadow") for k in self.world_progs)
        if self.props.bool_flag("shadow.enabled", has_shadow_prog) and has_shadow_prog:
            with self.timed("shadow"):
                self.safe("shadow pass", self.render_shadow, fs)
            if dump:
                self.dumper.texture("shadow_shadowtex0", self.shadowtex[0])
                self.dumper.texture("shadow_shadowtex1", self.shadowtex[1])
                for i, (m, a) in enumerate(self.shadowcolor):
                    self.dumper.texture(f"shadow_shadowcolor{i}", m)
            self.run_group("shadowcomp", U)
            for i in range(len(self.shadowcolor)):
                if self.shadowcolor_mip.get(i):
                    t = self.shadowcolor[i][1 if i in self.shadow_flipped_final else 0]
                    t.generate_mipmaps()
                    self.mip_enabled.add(t.id)
        if not prepare_first:
            self.run_group("prepare", U)
        # ---- gbuffers (opaque)
        self.depth["depthtex0"].clear((1.0, 0, 0, 0))
        if self.dh:
            self.depth["dhDepthTex0"].clear((1.0, 0, 0, 0))
        self.opaque_written = set()
        with self.timed("gbuffers_opaque"):
            self.safe("gbuffers (opaque)", self.render_opaque, fs)
        # beginHand: depthtex2 = depth before hand; beginTranslucents: depthtex1 = depth before translucents
        for k in ("depthtex2", "depthtex1"):
            GL.glCopyImageSubData(self.depth["depthtex0"].id, GL.GL_TEXTURE_2D, 0, 0, 0, 0,
                                  self.depth[k].id, GL.GL_TEXTURE_2D, 0, 0, 0, 0, self.w, self.h, 1)
        if dump:
            self._dump_gbuffers("gbuffers_opaque", self.flipped_after_prepare, self.opaque_written)
        self.run_group("deferred", U)
        # ---- translucents
        self.translucent_written = set()
        with self.timed("gbuffers_translucent"):
            self.safe("gbuffers (translucent)", self.render_translucent, fs)
        if dump:
            self._dump_gbuffers("gbuffers_translucent", self.flipped_after_translucent, self.translucent_written)
            self.dumper.depth("depthtex0", self.depth["depthtex0"], 0.05, U["far"])
            self.dumper.depth("depthtex1", self.depth["depthtex1"], 0.05, U["far"])
            if self.dh:
                self.dumper.depth("dhDepthTex0", self.depth["dhDepthTex0"], U["dhNearPlane"], U["dhFarPlane"])
        self.center_depth = float(self._read_center_depth())
        # ---- composite + final
        self.run_group("composite", U)
        with self.timed("final"):
            ctx = {"reads_alt": self.final.reads_alt, "stage": "final", "shadow_reads_alt": self.shadow_flipped_final}
            for c in self.final.computes:
                self.safe(c.name, self.dispatch, c, ctx, U, self.w, self.h)
            self.barrier()
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.final_fbo)
            GL.glViewport(0, 0, self.w, self.h)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glDisable(GL.GL_BLEND)
            GL.glDepthMask(GL.GL_FALSE)
            if self.final.frag is not None and self.final.frag.ok:
                for b in self.final.mip:
                    t = self.targets[b].alt if b in self.final.reads_alt else self.targets[b].main
                    t.generate_mipmaps()
                    self.mip_enabled.add(t.id)

                def _final():
                    self.bind(self.final.frag, ctx, U)
                    self.draw_quad()
                self.safe("final", _final)
            else:
                # no final pass: copy colortex0 (as read at this point) to the screen
                src = self.targets[0].alt if 0 in self.final.reads_alt else self.targets[0].main
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo([src], None))
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.final_fbo)
                GL.glBlitFramebuffer(0, 0, self.w, self.h, 0, 0, self.w, self.h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            # end-of-frame swap: copy alt -> main for buffers read from alt at the end (non-cleared ones)
            for b in self.swap:
                t = self.targets[b]
                GL.glCopyImageSubData(t.alt.id, GL.GL_TEXTURE_2D, 0, 0, 0, 0, t.main.id, GL.GL_TEXTURE_2D, 0, 0, 0, 0,
                                      t.w, t.h, 1)
        GL.glFinish()
        if dump:
            self._dump_images("end_of_frame")
            self._dump_ssbos()
        if self._npy_seq_on() and 0 in self.ssbos:
            # --npy-seq: the head of SSBO 0 (often a pack's global state, e.g. exposure) at the end of the frame
            b0, size0 = self.ssbos[0]
            a0 = np.zeros(min(size0, 256) // 4 * 4, np.uint8)
            GL.glGetNamedBufferSubData(b0, 0, len(a0), a0)
            os.makedirs(self.cfg.npy_seq_dir, exist_ok=True)
            np.save(os.path.join(self.cfg.npy_seq_dir, f"global_f{self.frame_index:03d}.npy"), a0.view(np.float32))
        self.first_frame = False

    def _read_center_depth(self):
        a = np.zeros(1, np.float32)
        GL.glGetTextureSubImage(self.depth["depthtex0"].id, 0, self.w // 2, self.h // 2, 0, 1, 1, 1,
                                GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, 4, a)
        return a[0]

    def read_output(self):
        a = np.zeros((self.h, self.w, 4), np.uint8)
        GL.glGetTextureImage(self.output.id, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, a.nbytes, a)
        return a[::-1]

    # ------------------------------------------------------------------ scene drawing
    def render_shadow(self, fs):
        U = fs.uniforms
        smv, sproj = fs.shadow_mv, fs.shadow_proj
        base = fs.draw_matrices("terrain", smv, sproj)
        common = {"entityId": -1, "blockEntityId": -1}
        stages = pack.STAGE_INDEX
        GL.glColorMask(True, True, True, True)
        P = self.props
        for layer, prog, st in (("solid", "shadow_solid", "TERRAIN_SOLID"),
                                ("cutout_mipped", "shadow_cutout", "TERRAIN_CUTOUT_MIPPED"),
                                ("cutout", "shadow_cutout", "TERRAIN_CUTOUT")):
            if not P.bool_flag("shadowTerrain", True):
                break
            pd = dict(base, renderStage=stages[st], **common)
            self.draw_world(prog, self.meshes.get(layer), U, pd, shadow=True, cull=False)
        if "entity" in self.meshes and P.bool_flag("shadowEntities", True):
            pd = dict(fs.draw_matrices("entity", smv, sproj), renderStage=stages["ENTITIES"],
                      entityId=fs.entity_id, blockEntityId=-1)
            self.draw_world("shadow_entities", self.meshes["entity"], U, pd, shadow=True, cull=False)
        GL.glCopyImageSubData(self.shadowtex[0].id, GL.GL_TEXTURE_2D, 0, 0, 0, 0, self.shadowtex[1].id,
                              GL.GL_TEXTURE_2D, 0, 0, 0, 0, self.shadow_res, self.shadow_res, 1)
        if P.bool_flag("shadowTranslucent", True):
            pd = dict(base, renderStage=stages["TERRAIN_TRANSLUCENT"], **common)
            self.draw_world("shadow_water", self.meshes.get("translucent"), U, pd, shadow=True, cull=False)
        self.barrier()

    def render_opaque(self, fs):
        U = fs.uniforms
        stages = pack.STAGE_INDEX
        proj = fs.proj
        written = set()
        common = {"entityId": -1, "blockEntityId": -1}
        # sky (vanilla: depth writes off)
        pd = dict(fs.draw_matrices("sky", fs.view, proj), renderStage=stages["SKY"], **common)
        if fs.draw_sky:
            # overworld: sky disc -> gbuffers_skybasic; End: textured end-sky box -> gbuffers_skytextured
            info = self.draw_world(fs.sky_program, self.meshes.get("sky"), U, pd, depth_mask=False, depth_test=False,
                                   cull=False)
            written.update(info.draw_buffers)
        if self.dh and self.cfg.dh_order == "before":
            written.update(self.render_dh_opaque(fs))   # DH renders its LODs before vanilla's solid layer
        base = fs.draw_matrices("terrain", fs.view, proj)
        for layer, prog, st in (("solid", "gbuffers_terrain_solid", "TERRAIN_SOLID"),
                                ("cutout_mipped", "gbuffers_terrain_cutout", "TERRAIN_CUTOUT_MIPPED"),
                                ("cutout", "gbuffers_terrain_cutout", "TERRAIN_CUTOUT")):
            pd = dict(base, renderStage=stages[st], **common)
            info = self.draw_world(prog, self.meshes.get(layer), U, pd)
            written.update(info.draw_buffers)
        if "entity" in self.meshes:
            pd = dict(fs.draw_matrices("entity", fs.view, proj), renderStage=stages["ENTITIES"],
                      entityId=fs.entity_id, blockEntityId=-1)
            info = self.draw_world("gbuffers_entities", self.meshes["entity"], U, pd)
            written.update(info.draw_buffers)
        if self.dh and self.cfg.dh_order == "after":
            written.update(self.render_dh_opaque(fs))
        self.opaque_written = written

    def dh_draw_uniforms(self, fs):
        """iris_* uniforms of a DH draw: camera-relative LOD vertices, gbufferModelView, the DH projection,
        identity texture matrices (Iris' DH transformer: gl_TextureMatrix[0/1] = mat4(1.0))."""
        pd = fs.draw_matrices("terrain", fs.view, fs.dh_proj)
        pd["iris_TextureMat"] = np.eye(4)
        pd["iris_LightmapTextureMatrix"] = np.eye(4)
        pd.update(renderStage=0, entityId=-1, blockEntityId=-1)
        return pd

    def render_dh_opaque(self, fs):
        """DH opaque LOD pass: dh_terrain into its colour targets with DH's own depth buffer (Iris
        DHCompatInternal: dhTerrainFramebuffer.addDepthAttachmentBypass(DH depth texture))."""
        m = self.meshes.get("dh_opaque")
        if m is None or m.count == 0 or not self.dh_progs:
            return set()
        with self.timed("dh_terrain"):
            info = self.draw_world("dh_terrain", m, fs.uniforms, self.dh_draw_uniforms(fs),
                                   depth_tex=self.depth["dhDepthTex0"])
        return set(info.draw_buffers)

    def render_translucent(self, fs):
        U = fs.uniforms
        stages = pack.STAGE_INDEX
        written = set()
        m = self.meshes.get("translucent")
        if m is not None:
            m.sort_back_to_front(fs.cam_region)
        pd = dict(fs.draw_matrices("terrain", fs.view, fs.proj), renderStage=stages["TERRAIN_TRANSLUCENT"],
                  entityId=-1, blockEntityId=-1)
        # vanilla also renders the water surface's backward (down-facing) side, so it is visible from below
        info = self.draw_world("gbuffers_water", m, U, pd, translucent=True, cull=False, default_blend=TRANSLUCENT_BLEND)
        written.update(info.draw_buffers)
        if self.dh and self.dh_progs:
            # DH deferred translucent pass: dhDepthTex1 = DH depth before it; culling off (Iris disables it)
            GL.glCopyImageSubData(self.depth["dhDepthTex0"].id, GL.GL_TEXTURE_2D, 0, 0, 0, 0,
                                  self.depth["dhDepthTex1"].id, GL.GL_TEXTURE_2D, 0, 0, 0, 0, self.w, self.h, 1)
            dm = self.meshes.get("dh_water")
            if dm is not None and dm.count:
                dm.sort_back_to_front(fs.cam_region)
                with self.timed("dh_water"):
                    info = self.draw_world("dh_water", dm, U, self.dh_draw_uniforms(fs), translucent=True, cull=False,
                                           default_blend=TRANSLUCENT_BLEND, depth_tex=self.depth["dhDepthTex0"])
                written.update(info.draw_buffers)
        rain = fs.rain_mesh
        if rain is not None and rain.count:
            pd = dict(fs.draw_matrices("terrain", fs.view, fs.proj), renderStage=stages["RAIN_SNOW"],
                      entityId=-1, blockEntityId=-1)
            info = self.draw_world("gbuffers_weather", rain, U, pd, translucent=True, depth_mask=False, cull=False,
                                   default_blend=TRANSLUCENT_BLEND)
            written.update(info.draw_buffers)
        self.translucent_written = written

    # ------------------------------------------------------------------ dumps
    def _dump_gbuffers(self, label, reads_alt, written):
        for b in sorted(written):
            t = self.targets[b].alt if b in reads_alt else self.targets[b].main
            self.dumper.texture(f"{label}_colortex{b}", t)

    def _dump_bound_images(self, p):
        """Images a pass can write (bound as image uniforms), except huge 3D volumes (dumped at frame end)."""
        names = set()
        for info in list(p.computes) + ([p.frag] if p.frag else []):
            if info.ok:
                names.update(n for n, *_ in info.prog.images if n in self.images)
        small = {n for n in names if self.images[n][0].w * self.images[n][0].h * self.images[n][0].d <= 4_000_000}
        if small:
            self._dump_images(p.name, only=small)

    def _dump_images(self, label, only=None):
        with self.dumper.timing():
            self._dump_images_(label, only)

    def _dump_images_(self, label, only=None):
        for name, (t, img) in self.images.items():
            if only is not None and name not in only:
                continue
            arr = t.read()
            kind = t.kind
            ch = t.info["channels"]
            fname = f"{self.dumper.seq:03d}_{label}_{name}.png"
            self.dumper.seq += 1
            st = dumps.stats(arr, ch)
            if t.target == GL.GL_TEXTURE_3D:
                if kind in ("uint", "int"):
                    # top-down map: value of the highest non-zero voxel per column, shaded by its height
                    nzm = arr[..., 0] != 0                        # (z, y, x)
                    H = nzm.shape[1]
                    top = H - 1 - np.argmax(nzm[:, ::-1, :], axis=1)   # (z, x)
                    has = nzm.any(axis=1)
                    val = np.take_along_axis(arr[..., 0], top[:, None, :], axis=1)[:, 0, :]
                    rgb, _ = dumps.visualize(val[..., None], kind, 1)
                    rgb = rgb * (0.35 + 0.65 * (top / max(1, H - 1)))[..., None]
                    rgb[~has] = 0.0
                    nz = int(nzm.sum())
                    extra = f" nonzero={nz} (top-down: highest voxel per column)"
                else:
                    sl = arr[arr.shape[0] // 2]
                    rgb, _ = dumps.visualize(sl, kind, ch)
                    extra = " (middle z slice)"
                    if arr.shape[0] <= 8:
                        # layered images (cascade arrays such as cloudBsmImg): every layer as an image + stats
                        for z in range(arr.shape[0]):
                            zname = fname.replace(".png", f"_z{z}.png")
                            zr, _ = dumps.visualize(arr[z], kind, ch)
                            dumps.save_rgb(self.dumper.path(zname), zr, flip=False)
                            self.log.info(f"    dump {zname:44s} {t.info['name']:8s} "
                                          f"{dumps.fmt_stats(dumps.stats(arr[z], ch))}")
                        extra = f" (middle z slice; layers 0..{arr.shape[0] - 1} in _z<k>.png)"
                dumps.save_rgb(self.dumper.path(fname), rgb, flip=False)
            else:
                rgb, _ = dumps.visualize(arr, kind, ch)
                dumps.save_rgb(self.dumper.path(fname), rgb)
                extra = ""
            self.log.info(f"    dump {fname:44s} {t.info['name']:8s} {dumps.fmt_stats(st)}{extra}")

    def _npy_texture(self, name):
        """Texture for an --npy name: colortexN (or a legacy name) as the next reader sees it after the frame
        (the side the final pass reads; non-cleared buffers left flipped are also copied there by the end-of-frame
        swap), a custom image by image or sampler name, a custom texture, depthtex0..2 / dhDepthTex0..1,
        shadowtex0..1 or shadowcolorN."""
        i = self._colortex_index(name)
        if i is not None:
            return self.targets[i].alt if i in self.final.reads_alt else self.targets[i].main
        if name in self.images:
            return self.images[name][0]
        if name in self.image_samplers:
            return self.image_samplers[name]
        if name in self.custom_textures:
            return self.custom_textures[name][0]
        if name in self.depth:
            return self.depth[name]
        if name in ("shadowtex0", "shadowtex1"):
            return self.shadowtex[int(name[-1])]
        if name.startswith("shadowcolor") and name[11:].isdigit() and int(name[11:]) < len(self.shadowcolor):
            k = int(name[11:])
            return self.shadowcolor[k][1 if k in self.shadow_flipped_final else 0]
        return None

    def dump_npy(self, names, out_dir):
        """--npy: raw read-back (glutil.Texture.read_rgba: float32 RGBA, GL row order, no flip; integer formats
        unconverted) of each named buffer at the end of the frame, saved as <out_dir>/<NAME>.npy."""
        self.log.section("npy dumps")
        for name in names:
            t = self._npy_texture(name)
            if t is None:
                self.log.warn(f"--npy {name}: no such colortex, image, custom texture or depth buffer (images: "
                              + ", ".join(sorted(self.images)) + ")")
                continue
            a = t.read_rgba()
            path = os.path.join(out_dir, f"{name}.npy")
            np.save(path, a)
            st = dumps.stats(a, 1 if a.ndim == 2 else 4)
            self.log.info(f"  {name}.npy {a.dtype} {'x'.join(map(str, a.shape))} {t.info['name']:8s} "
                          f"{dumps.fmt_stats(st)}")

    def _dump_ssbos(self):
        with self.dumper.timing():
            self._dump_ssbos_()

    def _dump_ssbos_(self):
        for idx, (b, size) in self.ssbos.items():
            a = np.zeros(size, np.uint8)
            GL.glGetNamedBufferSubData(b, 0, size, a)
            with open(self.dumper.path(f"ssbo{idx}.bin"), "wb") as f:
                f.write(a.tobytes())
            fl = a[: min(size, 256)].view(np.float32)
            ui = a[: min(size, 64)].view(np.uint32)
            self.log.info(f"    ssbo{idx} ({size} B) first floats: " + " ".join(f"{x:.4g}" for x in fl[:32]))
            self.log.info(f"    ssbo{idx} first uints: " + " ".join(str(int(x)) for x in ui[:16]), echo=False)
