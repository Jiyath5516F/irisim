# Contributing

Thanks for helping. irisim is useful exactly as far as it behaves like Iris, so most contributions are
"Iris does X here, irisim does Y".

## Setup

```
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
pytest                              # fast unit tests + one tiny end-to-end render (needs xvfb-run + Mesa)
```

## Guidelines

* **Follow Iris, and say where.** When you emulate a behaviour, name the Iris class / method it comes from
  in a comment or the commit message (e.g. `ProgramSet.readComputeArray`, `BufferFlipper`,
  `OptionAnnotatedSource`). If irisim deliberately differs, document it in the README's *Limits* and
  `docs/HOW_IT_WORKS.md`.
* **Stay pack-agnostic.** Nothing in `src/irisim/` may depend on a particular pack's option names, file
  layout, block ids or buffers. Pack-specific scenarios and cameras go in a preset JSON under
  `examples/presets/`.
* **Keep renders comparable.** Changes to the scene, meshing, textures or default uniforms change every
  render; call them out in `CHANGELOG.md`.
* **Log, don't crash.** A bad property line, an unknown sampler or a failing program should produce a
  `log.txt` warning and a rendered frame, not an exception.
* **Tests.** Add a unit test for parsing / preprocessing changes (`tests/test_options.py` style, no GL). For
  pipeline changes, extend `examples/minimal_pack` only if the smoke test stays fast (a few seconds).
* Python 3.9+, standard library + numpy, Pillow, PyOpenGL. Line length 120.

## Reporting a mismatch with the game

Include: the pack (or a minimal pack reproducing it), the irisim command line, `log.txt`, the relevant
`passes/*.png`, and a description or screenshot of what Iris shows in game, with the Iris and Minecraft
versions.
