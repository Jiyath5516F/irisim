# irisim — a headless mini-Iris for Minecraft shader packs

`irisim` runs a Minecraft Java shader pack's **real programs** (`gbuffers_*`, `shadow*`, `setup`,
`begin`, `prepare`, `deferred`, `composite`, `final`, compute passes, Distant Horizons programs) in
**Iris order** on a software OpenGL 4.5 *compatibility* context (Mesa llvmpipe under Xvfb), against a
**synthetic Minecraft-like voxel world**, and writes the final image, every intermediate buffer, per-pass
timings and NaN/Inf statistics.

It lets you (or an AI coding agent) look at what a shader change does **without starting Minecraft**:
on a CI box, over SSH, in a container, on a machine without a GPU.

```
irisim path/to/MyPack --scenario noon                    # -> irisim-out/noon/final.png + passes/ + log.txt
irisim path/to/MyPack.zip --scenario sunset --width 640 --height 360 --frames 4
```

It is *not* Minecraft and not Iris: it is a small re-implementation of the parts of Iris that decide what a
pack's programs see (sources, options, directives, buffers, flips, uniforms, draw order), driving a scene
built in Python. See [Limits](#limits).

Built with [Claude Code](https://claude.com/claude-code). Originally written as an in-tree tool of the
Vibe Shader pack and made generic here.

**License:** GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Features

* **Any Iris pack**: give it the pack folder, its `shaders/` folder or the `.zip`. Programs are discovered
  from the files on disk per dimension (`world0`, `world-1`, `world1`, falling back to the root) with
  Iris' fallback chain (`gbuffers_terrain -> gbuffers_textured_lit -> gbuffers_textured -> gbuffers_basic`,
  `shadow_* -> shadow`, ...), compute arrays (`prepare1.csh`, `prepare1_a.csh` ..), `program.*.enabled`.
* **Iris preprocessing**: `#include` resolution (absolute `/lib/x.glsl` and relative) with a line map, so
  compile errors read `lib/lighting.glsl:123`; Iris standard macros (`MC_VERSION`, `IRIS_VERSION`,
  `MC_RENDER_STAGE_*`, `IRIS_FEATURE_*`, `DH_BLOCK_*`, ...); `#extension` hoisting; compatibility patching
  of `gl_Vertex`, `gl_ModelViewMatrix`, `ftransform()`, ... like Iris' transformers.
* **Options like Iris**: options are discovered the way Iris' `OptionAnnotatedSource` does it and can be
  overridden from the command line (`--set NAME=VALUE`, `--on NAME`, `--off NAME`); `--list-options`
  prints them.
* **shaders.properties** (preprocessed with the option values): custom images, SSBOs (`bufferObject.N`),
  custom textures (PNG and raw `.bin` 1D/2D/3D with `.mcmeta` filtering), `scale.*`, `size.buffer.*`,
  `blend.*`, `flip.*`, `program.*.enabled`, `uniform.*` / `variable.*` custom uniforms (common subset of
  the expression language incl. `smooth`), `shadow.enabled`, `separateAo`, ...; `block.properties`
  (incl. block states and `layer.*` render-type overrides), `item.properties`, `entity.properties`.
* **Render targets**: `colortex0-15` with Iris' `BufferFlipper` semantics, formats / clears / mipmaps from
  the directives, `depthtex0/1/2`, `shadowtex0/1`, `shadowcolor0-N`, custom images (2D / 3D / relative),
  hardware shadow compare samplers.
* **Uniforms**: the common Iris / OptiFine uniform set (matrices, previous-frame matrices, celestial
  positions with `sunPathRotation`, shadow matrices with `shadowIntervalSize` snapping, weather, biome,
  time, eye brightness, `centerDepthSmooth`, `renderStage`, ...), plus the pack's custom uniforms.
* **A synthetic world**: 96 x 48 x 96 blocks of terrain, a lake, trees, a house with glass windows, a cave
  with torches, lava, flowers and grass, an entity; Nether and End variants; optional test structures
  (glass pieces, lamp posts, coloured-light booths, a closed room for GI tests). Meshed like Sodium (face
  culling, smooth lighting, vanilla AO, `mc_Entity`, `at_tangent`, `at_midBlock`, `mc_midTexCoord`).
* **Distant Horizons emulation** (`--dh`): `dh_terrain` / `dh_water` with the DH transformer's inputs and
  uniforms, an LOD terrain ring around the scene, `dhDepthTex0/1`.
* **Diagnostics**: per-pass PNG dumps of every buffer a pass wrote, per-channel min / max / mean and
  **NaN / Inf counts** for every dump, per-program GPU-side timings (llvmpipe), raw `.npy` dumps of any
  buffer, per-frame `.npy` sequences for flicker / convergence studies, uniforms or samplers a program
  uses that irisim does not provide.

## Install

irisim needs Linux with Mesa's llvmpipe (OpenGL 4.5 compatibility profile, GLX), Xvfb and Python 3.9+.

```
# Debian / Ubuntu
sudo apt install xvfb libgl1-mesa-dri libglx-mesa0 libx11-6 python3-venv
# Fedora
sudo dnf install xorg-x11-server-Xvfb mesa-dri-drivers mesa-libGL libX11

python3 -m venv .venv && . .venv/bin/activate
pip install .            # or: pip install -e ".[test]" for development
irisim --version
```

Python dependencies (installed by pip): `numpy`, `Pillow`, `PyOpenGL`. No `moderngl`, no GPU, no Java.
Developed with Mesa 25.2 (LLVM 20) and Python 3.13; any Mesa whose llvmpipe reports
`4.5 (Compatibility Profile)` should work (`glxinfo -B` under `xvfb-run` shows it).

Running from a checkout without installing: `PYTHONPATH=src python3 -m irisim ...`.

## Usage

```
irisim PACK [options]
```

`PACK` is a pack folder (containing `shaders/`), the `shaders/` folder itself, or a `.zip`. Without a
`DISPLAY`, irisim re-runs itself under `xvfb-run` (`--xvfb auto|always|never`).

```
irisim MyPack --scenario noon                                  # one preset
irisim MyPack --scenario all --frames 4 --dump none            # every default scenario (noon sunset night cave underwater rain interior nether end)
irisim MyPack --list-scenarios                                 # presets and cameras
irisim MyPack --list-options                                   # options the pack declares (file, kind, default, values)
irisim MyPack --scenario cave --set SHADOW_QUALITY=2 --off BLOOM --on DEBUG_GI
irisim MyPack --time 13000 --camera overview --rain 1 --thunder 1 --name storm
irisim MyPack --pos 40,36,60 --yaw -90 --pitch 10 --time 1000  # free camera (scene block coordinates)
irisim MyPack --scenario noon --frames 12 --motion 0.2         # camera moving forward: TAA / reprojection tests
irisim MyPack --scenario noon --dh --frames 12                 # Distant Horizons LOD ring
irisim MyPack --scenario noon --npy colortex3,depthtex0        # raw float buffers for numpy analysis
irisim MyPack --scenario noon --npy-seq composite_colortex0 --frames 16   # per-frame dumps (flicker)
irisim MyPack --presets examples/presets/test-structures.json --scenario gallery   # coloured-light booths
irisim MyPack --scenario rain --precipitation snow --temperature 0.0       # snowy biome uniforms
irisim MyPack --scenario noon --gl-vendor AMD --mc-version 12004           # other host macros
```

`irisim --help` lists every option. The most used ones:

| option | meaning |
|---|---|
| `--scenario S` | preset: time, camera, weather, world (`--list-scenarios`); `all` = the default set |
| `--presets FILE` | load more cameras / scenarios from JSON (see `examples/presets/`) |
| `--width/--height` | default 480x270 |
| `--frames N` | frames to render (default 8; temporal passes need a few frames to converge) |
| `--time T` `--time-speed` | `worldTime` ticks (1000 morning, 6000 noon, 12500 sunset, 18000 midnight) |
| `--rain R` `--thunder T` `--wetness W` `--rain-ramp START,FRAMES` | weather uniforms (+ rain quads with `gbuffers_weather`) |
| `--precipitation none\|rain\|snow` `--temperature` `--rainfall` `--fog-color R,G,B` | biome uniforms |
| `--world world0\|world-1\|world1` | dimension (scene variant follows) |
| `--camera C` / `--pos x,y,z --yaw --pitch --fov` | camera preset or free camera |
| `--underwater` | force `isEyeInWater = 1` |
| `--set/--on/--off` | option overrides (rewritten in the declaring file, like Iris) |
| `--define NAME[=V]` | extra macro for every program and the `.properties` files |
| `--shadow-res N` | shadow map size, applied through the pack's `shadowMapResolution` option (default 1024) |
| `--t0 S` `--dt S` | `frameTimeCounter = (t0 + frame * dt) mod 3600` (test late-in-the-hour animation) |
| `--held ITEM` | `heldItemId` / `heldBlockLightValue` |
| `--pbr` | generate labPBR `_n` / `_s` atlases instead of Iris' flat defaults |
| `--dh` `--dh-distance N` | Distant Horizons emulation |
| `--dump last\|all\|none` `--save-frames` | per-pass dumps of the last / every / no frame; every frame's final image |
| `--npy NAME,..` `--npy-seq PASS_colortexN,..` | raw numpy dumps (see below) |
| `--log-uniforms NAME,..` | log these uniforms (built-in or custom) every frame |
| `--out DIR` `--name N` | output folder (default `./irisim-out/<scenario>`) |
| `--vertex-space sodium\|chunk` | Sodium (camera-relative `gl_Vertex`, default) or chunk-relative vertices |

## Output

`<out>/<name>/`:

| file | contents |
|---|---|
| `final.png` | the final pass output of the last frame |
| `log.txt` | GL renderer, option overrides, every program compiled (ms, render targets, compute work groups), directives (formats, clears, shadow settings), the flip schedule, resources, per-frame time and mean colour, **every dump with statistics**, timings table, missing uniforms / samplers, summary |
| `passes/NNN_<pass>_<buffer>.png` | every buffer each pass wrote, in execution order (last frame; `--dump all`: every frame) |
| `passes/*_a.png` | the alpha channel, when it is not constant |
| `passes/NNN_shadow_shadowtex0.png` ... | shadow depth (normalised to its occupied range) and `shadowcolorN` |
| `passes/NNN_depthtex0.png` | linearised scene depth |
| `passes/NNN_end_of_frame_<image>.png` | custom images (3D: middle slice; integer: hashed colours / height map) |
| `passes/ssbo<N>.bin` | raw SSBO contents |
| `<NAME>.npy` | `--npy NAME`: raw buffer after the last frame, `(h, w, 4)` float32 (or uint32 / int32), GL row order (row 0 = bottom) |
| `seq/<pass>_colortex<N>_fNNN.npy`, `seq/final_fNNN.npy` | `--npy-seq`: per-frame dumps |
| `failed_sources/<prog>.<stage>.glsl` | numbered, fully preprocessed source of a program that failed to compile |

**Reading the dumps.** Float buffers are Reinhard-tonemapped and sRGB-encoded for the PNG (NaN shows as
magenta, +Inf as cyan); unorm as is; snorm `*0.5+0.5`; integer formats as hashed colours. Every dump line
in `log.txt` looks like

```
    dump 031_deferred1_colortex3.png   RGBA16F  min (0, 0, 0, 1) max (12.5, 9.1, 7.7, 1) mean (0.41, 0.39, 0.35, 1)  !! NaN=12 Inf=0
```

and a `!! NaN=` marker is the quickest way to find the first pass that poisons a history buffer.

**Timings.** Each pass is bracketed with `glFinish`, so the table at the end of `log.txt` is the llvmpipe
cost per program (frame 0 includes shader JIT; the `mean1+` column averages the later frames). CPU
rendering is orders of magnitude slower than a GPU, but the *ratios* between passes and A/B comparisons
of one pass are informative. `--no-timing` drops the `glFinish`es.

The exit status is 0 on success and 2 when a program failed to compile (the frame is still rendered:
failed fullscreen passes clear their outputs, failed gbuffers programs skip their draws).

## Limits

irisim approximates the game; trust it for "does this compile, run, and look roughly right", for buffer
contents and for NaN hunting, not for final colour grading.

* No hand, particles, block entities, vanilla clouds / sun / moon geometry, beacon beams, glint, damaged
  blocks, lines or translucent entities: those programs are compiled and reported, never drawn.
* No frustum or shadow culling; translucent terrain is drawn with culling off (Minecraft culls glass back
  faces), so translucent programs see back faces the game never draws.
* Vanilla sky and fog colours are approximations; `eyeBrightnessSmooth` and `wetness` take their converged
  values unless `--wetness` / `--rain-ramp`; no `biome` / `biome_category` uniforms or `BIOME_*` macros (use
  `--temperature`, `--rainfall`, `--precipitation`, `--fog-color`); the weather quads always use the rain
  texture; `is_sneaking` is always false.
* Custom uniform expressions cover the common functions; indirect dispatch, `alphaTest.*`,
  `texture.<stage>.*` overrides, `customTexture` resource locations (`minecraft:...`), block tags (`%tag`)
  and profiles are not supported.
* Memory barriers are issued around *every* pass, so missing barriers that a real driver punishes go
  unnoticed. Image/sampler limits of real drivers (e.g. NVIDIA's 8 image uniforms per stage) are not
  enforced: use a validator such as `glslangValidator` plus the driver limits for that.
* With more than one llvmpipe thread, passes that race by design give run-to-run differences; for exact
  before / after comparisons run with `LP_NUM_THREADS=1`.
* llvmpipe is slow: a heavy pack takes seconds per frame at 640x360. Keep renders small (see the agent
  [skill](skills/irisim/SKILL.md) for a render budget).

[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) describes what is emulated and how.

## Running on a shared machine

llvmpipe uses `LP_NUM_THREADS` threads (irisim defaults it to the core count minus one, at most 8). When
several people or agents render on one machine, serialise renders with a lock, and always with `flock -o`:

```
flock -o /tmp/irisim.lock irisim MyPack --scenario noon --out /tmp/me/out
```

Without `-o` the lock's file descriptor is inherited by `xvfb-run` and Xvfb; if a render is interrupted,
the orphaned Xvfb keeps holding the lock and every later `flock` waits forever.

## For AI agents

[AGENTS.md](AGENTS.md) and the Claude Code skill [skills/irisim/SKILL.md](skills/irisim/SKILL.md) explain
how an agent should use irisim to iterate on a shader without the game: commands, reading `log.txt` and the
dumps, and a render budget. Install the skill with
`mkdir -p ~/.claude/skills && cp -r skills/irisim ~/.claude/skills/` (or into a project's `.claude/skills/`).

## Development

```
pip install -e ".[test]"
pytest                     # unit tests + an end-to-end render of examples/minimal_pack (needs xvfb-run)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md). `examples/minimal_pack` is a
tiny pack (terrain with sun shadows, sky, fog, vignette, a few options) that is also a good starting point
for experiments.
