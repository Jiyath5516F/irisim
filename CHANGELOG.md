# Changelog

## 0.1.0 — first standalone release

Extracted from the in-tree `tools/irisim` of the Vibe Shader pack and made generic. Built with
Claude Code.

* Installable package (`pip install .`) with an `irisim` command and `python -m irisim`.
* Works with any Iris pack: `irisim PACK` takes a pack folder, its `shaders/` folder or a `.zip`.
* Option discovery reimplemented after Iris' `OptionAnnotatedSource` (`options.py`): every shader file is
  scanned; boolean `#define`s count only when referenced by `#ifdef` / `#ifndef`; const options only for
  Iris' configurable const directives. Overrides are rewritten in whichever file declares the option
  (previously only in the pack's own settings folder, through the pack's validation script).
* Camera and scenario presets moved to JSON (`src/irisim/presets/default.json`); `--presets FILE` loads
  more. The test-structure scenarios and cameras (glass, village, gallery, poles, hall, furnace, deepcave)
  moved to `examples/presets/test-structures.json`; the structures themselves remain as flags.
* `--scenario all` is handled by the command itself (one process per scenario).
* irisim re-runs itself under `xvfb-run` when no `DISPLAY` is set (`--xvfb auto|always|never`);
  `LP_NUM_THREADS` defaults to the core count minus one (at most 8).
* New: `--list-options`, `--list-scenarios`, `--define NAME[=VALUE]`, `--mc-version`, `--gl-vendor`, `--os`,
  `--log-uniforms`, `--version`; the log header names the irisim version and the pack path.
* Removed: the `IRISIM_*` environment-variable defaults of the test-structure and `--npy-seq` options (use
  the flags); the pack-specific uniform names in the per-frame weather line (use `--log-uniforms`).
* Default output folder is `./irisim-out/<name>`.
* `examples/minimal_pack`: a tiny pack (terrain with sun shadows, sky, fog, vignette, options) used by the
  smoke test.
* Docs: README, `docs/HOW_IT_WORKS.md`, an agent skill (`skills/irisim/SKILL.md`), `AGENTS.md`,
  `CONTRIBUTING.md`.

Rendering, the scene and the pipeline are unchanged from the in-tree version, so renders of the same
pack with the same flags should match it (up to llvmpipe's multi-threading noise).
