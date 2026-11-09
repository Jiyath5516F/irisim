# Examples

## `minimal_pack/`

A tiny Iris pack (about 150 lines of GLSL): terrain with sun shadows from `shadowtex0`, a sky pass, a
composite with distance fog and a debug-normals switch, a final pass with a simple tonemap and a vignette.
It declares a few options of every kind (boolean `#define`s, value `#define`s, const directives) in
`shaders/lib/settings.glsl`, so it shows the option machinery, and it is what `tests/test_smoke.py`
renders.

```
irisim examples/minimal_pack --list-options
irisim examples/minimal_pack --scenario noon --width 640 --height 360 --frames 2
irisim examples/minimal_pack --scenario sunset --on DEBUG_NORMALS --name normals
irisim examples/minimal_pack --scenario noon --set FOG_DENSITY=2.0 --set sunPathRotation=0.0 --name foggy
```

Copy it to start experimenting with a new effect: every pass it has is dumped to `passes/`.

## `presets/test-structures.json`

Cameras and scenarios for irisim's optional test structures, for lighting and glass work on any pack:

| scenario | what | structure flag |
|---|---|---|
| `glass` | clear / stained panes, a thick clear glass cluster, white / orange / tinted glass, ice, honey, slime (cameras `glass`, `glass2`, `glass3`, `glass4`, `interior2`) | `--glass-gallery` |
| `village` | 8 lamp posts with lanterns and torches at night | `--village` |
| `gallery` | 9 white booths, one emitter each (torch, soul torch, redstone torch, lantern, soul lantern, sea lantern, glowstone, redstone lamp, shroomlight); change with `--gallery-kinds` | `--gallery` |
| `poles` | 5 lights on poles against the night sky (add `--rain 1` for fog glow); `--pole-kinds` | `--poles` |
| `hall` | closed 30 x 7 x 30 hall lit by 4 floor emitters (block-light GI over distance) | `--hall` |
| `furnace` | closed 7 x 5 x 7 white room with one emitter (GI energy conservation) | `--furnace` |
| `deepcave` | the cave with the whole scene 40 blocks lower (world y ~27), for altitude-based cave logic | `--origin-dy -40` |

```
irisim MyPack --presets examples/presets/test-structures.json --scenario gallery --width 960 --height 540 --frames 12
irisim MyPack --presets examples/presets/test-structures.json --scenario glass --camera glass2
```

A preset file is JSON with `cameras` (`eye` and `target` in scene block coordinates; an eye `y` of `null`
stands on the ground), `scenarios` (`time`, `camera`, `rain`, `thunder`, `world`, `underwater`,
`structures`, `origin_dy`) and optionally `all` (the list `--scenario all` renders). Write your own for your
pack's regression views.

## Pack-specific workflows

The in-tree version of irisim was developed against the Vibe Shader pack. Typical commands from that
project, as an example of using irisim in a pack's own workflow:

```
irisim Vibe-Shader --scenario noon --frames 4 --set GI_RAYS=2 --off TAA
irisim Vibe-Shader --scenario noon --set DEBUG_VIEW=6                 # a pack debug view
irisim Vibe-Shader --scenario cave --held torch --test-step --pos 69.3,34.27,85.5 --yaw -90 --pitch 12 --time 18000
irisim Vibe-Shader --presets examples/presets/test-structures.json --scenario furnace --set CACHE_UPDATE_RATE=1 \
       --npy-seq deferred9_colortex0                                   # GI convergence per frame
```

(`GI_RAYS`, `TAA`, `DEBUG_VIEW`, `CACHE_UPDATE_RATE` are that pack's options; `irisim PACK --list-options`
shows yours.)
