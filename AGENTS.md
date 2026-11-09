# Notes for AI agents

* **Using irisim** to test a shader pack: follow [skills/irisim/SKILL.md](skills/irisim/SKILL.md) (a Claude
  Code Agent Skill; other agents can read it as plain instructions). It covers when to render, the
  commands, the edit / render / compare loop, how to read `log.txt` and the pass dumps, the render budget
  and the `flock -o` lock on shared machines.
* **Working on irisim itself**: read [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) and
  [CONTRIBUTING.md](CONTRIBUTING.md). The code is in `src/irisim/`; run `pytest` (the smoke test renders
  `examples/minimal_pack` at 160x90 and needs `xvfb-run`). Keep irisim pack-agnostic: no option names,
  file layouts or ids of a particular pack in `src/`; pack-specific presets belong in `examples/`.
* Keep test renders small (640x360 or less, a few frames) and delete their output.
