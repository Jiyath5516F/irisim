---
name: irisim
description: Render a Minecraft Java (Iris / OptiFine-style) shader pack headlessly, without the game, to check that a shader change compiles, runs and looks right; inspect intermediate buffers, find NaN/Inf, compare before/after, time passes. Use when working on GLSL in a shader pack's shaders/ folder (gbuffers_*, deferred, composite, final, shadow, compute programs, shaders.properties) and you need to see the result.
---

# irisim: iterate on a shader pack without opening Minecraft

irisim runs the pack's real programs in Iris order on Mesa llvmpipe (software OpenGL 4.5 compatibility,
under Xvfb) against a synthetic voxel world, and writes `final.png`, one PNG per buffer per pass, and a
`log.txt` with compile errors, statistics and timings. It is an approximation of the game (no particles,
hand, block entities or vanilla clouds; vanilla sky/fog colours approximated), good for "does it compile,
run, and look roughly right", buffer debugging and NaN hunting, not for final colour grading.

## When to use

* After editing any program, include file or `shaders.properties`: confirm it compiles and renders.
* A user reports black / white / magenta / flickering output, a broken pass, banding, NaN spread.
* Before / after comparisons of a visual change, or timing a pass after an optimisation.
* Checking an option (`--set`, `--on`, `--off`), a dimension (`--world world-1`), weather, time of day,
  underwater, Distant Horizons (`--dh`).

Do not use it to judge exact colours against an in-game screenshot; the user's game is the reference.

## Commands

```
irisim PACK --list-options                         # options as Iris discovers them (file, kind, default, values)
irisim PACK --list-scenarios                       # presets: noon morning sunset night cave underwater rain interior nether end
irisim PACK --scenario noon --width 640 --height 360 --frames 4 --out OUT --name noon_a
irisim PACK --scenario night --set SHADOW_RES=2048 --off BLOOM --on DEBUG_VIEW --name test
irisim PACK --pos 40,36,60 --yaw -90 --pitch 10 --time 1000 --name custom_view
irisim PACK --scenario noon --npy colortex3 --dump none        # raw float buffer -> OUT/NAME/colortex3.npy
irisim PACK --scenario noon --frames 12 --npy-seq composite_colortex0   # per-frame dumps for flicker
```

`PACK` is the pack folder, its `shaders/` folder or a `.zip`. Without `irisim` on PATH, run from a
checkout with `PYTHONPATH=src python3 -m irisim`. Exit status 2 means a program failed to compile.

## Iteration loop

1. **Baseline first.** Before editing, render the view that shows the problem (same flags you will use
   after): `--name before`. If you only need to know whether it compiles, use `--frames 1 --dump none`.
2. Edit the shader.
3. Render with identical flags and `--name after`.
4. Read `OUT/after/log.txt` (see below), then look at `final.png` and the relevant `passes/*.png` of both
   runs. Compare numerically when the change is subtle:
   `python3 -c "import numpy as np; from PIL import Image as I; a=np.asarray(I.open('OUT/before/final.png'),float); b=np.asarray(I.open('OUT/after/final.png'),float); print(abs(a-b).mean(), abs(a-b).max())"`
5. For bit-exact before/after comparisons set `LP_NUM_THREADS=1` for both runs (multi-threaded llvmpipe
   reorders racy work and gives small run-to-run differences).
6. Delete render folders you no longer need (`rm -rf OUT/before`): dumps are large.

To compare against an older version of the pack without touching the working tree, copy the old
`shaders/` folder somewhere (for example from `git show` / `git worktree` output you already have) and pass
that folder as `PACK`.

## Reading log.txt

Sections, in order: header (GL renderer, pack path, scenario), `== pack` (option overrides, .properties),
`== programs` (one line per compiled program; **errors appear here as `file:line` with the driver
message**), `== directives` (buffer formats, clears, shadow settings), `== schedule` (flip schedule),
`== scene`, `== frames`, `== timings in ms`, missing uniforms / samplers, `== summary`.

* `grep -n "error\|failed\|!! NaN\|warn" OUT/NAME/log.txt` is the first thing to run.
* A compile error points at the original file and line (`lib/lighting.glsl:123`); the full preprocessed
  source of a failed program is in `OUT/NAME/failed_sources/<program>.<stage>.glsl` with line numbers.
* Each dump line: `dump NNN_<pass>_<buffer>.png FORMAT min (...) max (...) mean (...)` and, when present,
  `!! NaN=N Inf=M`. The **first** pass in execution order with a NaN marker is where to look; later
  passes usually just inherit it (history buffers spread NaN over frames).
* `uniforms declared by programs but not provided by irisim`: those stay 0 in irisim. If a pass depends
  on one, its output in irisim is not representative.
* `samplers/images without a resource`: bound to a black dummy; usually a name typo or a feature irisim
  does not emulate.
* Timings: frame 0 includes shader JIT; use the `mean1+` column. Compare one pass before/after, not
  absolute numbers (llvmpipe is a CPU).

## Reading the dumps

`passes/NNN_<pass>_<buffer>.png` are numbered in execution order (shadow, gbuffers_opaque,
gbuffers_translucent, deferredN, compositeN, final). Float buffers are tone-mapped for display (Reinhard +
sRGB), so very bright HDR values all look white: use the logged max or `--npy` for actual values. NaN is
magenta, +Inf cyan. `*_a.png` is the alpha channel. `NNN_depthtex0.png` is linearised depth;
`shadow_shadowtex0.png` the shadow map. Use the Read tool on PNGs to view them; prefer the one or two
passes that matter over viewing all of them.

For numbers, load `--npy` dumps: arrays are `(h, w, 4)` float32 in GL row order (row 0 is the *bottom*
row), integer formats unconverted.

## Render budget

llvmpipe is slow and dumps are big. Keep it small:

* Default to **640x360** (or the 480x270 default). Never render 1080p unless the user asks; 960x540 at
  most for fine detail.
* **4 frames** is enough for most checks; temporal effects (TAA, accumulation, exposure) need 8-12.
  `--frames 1 --dump none` for a compile/run check.
* Render only the scenarios that show what you changed (e.g. `night` for a moon or aurora change, `rain`
  for wetness, `cave` for block light, `underwater`, `nether`, `end`). Avoid `--scenario all` unless the
  change is global; with it, use `--frames 2 --dump none`.
* `--dump none` when you only need `final.png`; `--dump last` (default) when you need the passes.
* Delete output folders when done.

## Shared machines: use a lock

When other agents or jobs may render on the same machine, serialise renders with a lock file, and
always pass `-o`:

```
flock -o /path/to/irisim.lock irisim PACK --scenario noon --out OUT --name x
```

Without `-o`, an interrupted render leaves an orphaned Xvfb holding the lock forever. Use the lock path the
project or user gives you; one render at a time per lock.

## Gotchas

* The scene is synthetic: block ids come from the pack's `block.properties` by vanilla block name
  (`log.txt` lists blocks with no id). Shaders that key effects on ids see the pack's real ids.
* Options are rewritten in the file that declares them, like Iris. `--set` on an undeclared name warns
  and does nothing. Check names with `--list-options`.
* `--shadow-res` (default 1024) overrides the pack's `shadowMapResolution` option; pass `--shadow-res 0`
  to keep the pack's value.
* Not emulated: particles, hand, block entities, vanilla clouds/sun/moon, beacon beams, glint, entities
  other than one box. Driver limits (e.g. NVIDIA's 8 image uniforms per stage, texture unit counts) are
  not enforced: a clean irisim run does not prove a pack loads on every GPU.
* More details: the project's `README.md` (options table, output files) and `docs/HOW_IT_WORKS.md`.
