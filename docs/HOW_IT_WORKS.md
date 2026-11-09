# How irisim works

irisim is a small re-implementation of the parts of Iris (`net.irisshaders.iris`) that decide what a
shader pack's programs see, on top of raw OpenGL through PyOpenGL. Everything runs in one Python process
on a Mesa llvmpipe context. This page follows a frame from the pack on disk to `final.png`.

```
 PACK ──► options.py ──► pack.py ─────────────► pipeline.py ──────────────► dumps.py
          (discover,     (includes, macros,      (resources, flip schedule,   (PNG / npy dumps,
           overrides)     patching, directives,   frame order, uniforms,       statistics)
                          .properties, id maps)   draws, dispatches)
                                                      ▲
 scene.py / textures.py ──────────────────────────────┘  (voxel world, Sodium-like meshes, atlas, lightmap)
 camera.py  (matrices, celestial angles, shadow matrices)   cli.py  (arguments, presets, per-frame uniforms)
 glctx.py   (GLX 4.5 compatibility context)                 customuniforms.py (uniform.* / variable.*)
```

| module | role |
|---|---|
| `cli.py` | command line, presets, the `Driver` (scene setup, per-frame uniform state, frame loop, report) |
| `glctx.py` | creates an OpenGL 4.5 **compatibility** context with `glXCreateContextAttribsARB` on a 16x16 pbuffer (Xvfb display); everything renders into irisim's own framebuffer objects |
| `options.py` | option discovery and rewriting (Iris `OptionAnnotatedSource`), Iris standard macros |
| `pack.py` | sources, includes, line map, compatibility patching, directive parsing, `.properties` parsing, id maps, program discovery |
| `cpp.py` | a small C preprocessor (for `.properties` files and for extracting the *active* text of a program) |
| `pipeline.py` | the mini Iris pipeline: programs, render targets, flips, shadow / gbuffers / composite passes |
| `glutil.py` | textures, sampler objects, program compile / link / reflection, uniform upload |
| `customuniforms.py` | the `uniform.*` / `variable.*` expression language |
| `scene.py`, `textures.py` | the synthetic world, its meshes, the procedural block atlas and the vanilla lightmap |
| `camera.py` | view / projection / shadow matrices, vanilla celestial angle |
| `dumps.py` | buffer visualisation, statistics, timing |

## 1. Context

Iris packs are written against `#version 1xx/3xx/4xx compatibility` and use fixed-function built-ins
(`gl_Vertex`, `gl_ModelViewMatrix`, `gl_FragData`). Mesa's EGL offers only core profiles for desktop GL,
so irisim uses GLX with the compatibility-profile bit under Xvfb. `cli.main` re-executes itself under
`xvfb-run -a` when `DISPLAY` is not set (`IRISIM_IN_XVFB=1` marks the child). `LP_NUM_THREADS` sets the
llvmpipe thread count.

## 2. Options

`options.discover_options()` scans every shader source file of the pack like Iris'
`OptionAnnotatedSource`:

* `#define NAME` / `//#define NAME` (optionally with a trailing `//` comment) is a **boolean** option only
  if `NAME` appears on an `#ifdef NAME` or `#ifndef NAME` line somewhere (Iris does not count `#if defined`);
* `#define NAME VALUE // [a b c]` is a **value** option (the bracketed allowed-values list is required);
* `const int|float|bool NAME = VALUE; // [a b c]` is an option only for Iris' configurable const
  directives (`shadowMapResolution`, `shadowDistance`, `sunPathRotation`, `*Halflife`, ...).

`--set NAME=VALUE`, `--on NAME`, `--off NAME` are applied by rewriting the declaring lines while the
sources are read, which is how Iris applies a changed option: a boolean is commented or uncommented, a
value replaced. `--shadow-res N` (default 1024) becomes a `shadowMapResolution` override when the pack
declares that option, so the shader constant and the shadow map size agree. The enabled booleans and the
values are also macros for the `.properties` preprocessor.

## 3. Sources

For each program stage (`pack.load_stage`), `#include "/abs/path.glsl"` (relative to `shaders/`) and
`#include "rel.glsl"` (relative to the including file) are inlined recursively, keeping a line map
`(file, line)` for every line. `pack.build_compile_text` then

1. keeps `#version` first and hoists every `#extension` line after it;
2. inserts the Iris standard macros (`MC_VERSION`, `IRIS_VERSION`, `MC_GL_VENDOR_*`, `MC_OS_*`,
   `MC_RENDER_STAGE_*`, `IRIS_FEATURE_*`, `DH_BLOCK_*`, `DISTANT_HORIZONS` with `--dh`) and `--define`s;
3. patches the compatibility built-ins the way Iris' transformers do:
   * gbuffers / shadow programs: `gl_Vertex`, `gl_Normal`, `gl_Color`, `gl_MultiTexCoord0..3`,
     `gl_ModelViewMatrix(+Inverse)`, `gl_ProjectionMatrix(+Inverse)`, `gl_ModelViewProjectionMatrix`,
     `gl_NormalMatrix`, `gl_TextureMatrix[0..2]`, `ftransform()` become `iris_*` attributes / uniforms.
     `mc_Entity` (location 5), `mc_midTexCoord` (6), `at_tangent` (7), `at_midBlock` (8) are generic
     attributes. In the default `--vertex-space sodium`, `gl_Vertex` is camera-relative (Sodium adds a
     region offset, `iris_RegionOffset`) and `gl_ModelViewMatrix` is the view rotation;
   * composite / final programs get a fullscreen quad (`gl_Vertex.xy` and `gl_MultiTexCoord0` in 0..1),
     an orthographic projection to NDC, identity texture matrices and white `gl_Color`;
   * `dh_*` programs additionally get the DH transformer's `int dhMaterialId` and lose `gl_Color` in the
     fragment stage, as in Iris;
4. records which declarations the patch needs and injects them.

Patching never changes the line count, so a GL compile error `0:123` maps back to `lib/x.glsl:45` in
`log.txt`; a failing program's full text goes to `failed_sources/`.

**Directives** are read from the *active* text of the fragment (or compute) stage: `cpp.MiniCpp` removes
inactive `#if` branches but keeps comments, as JCPP does in Iris. From it irisim reads `RENDERTARGETS` /
`DRAWBUFFERS` (last occurrence wins), `const` directives (`colortexNFormat`, `colortexNClear`,
`colortexNClearColor`, `colortexNMipmapEnabled`, `shadowMapResolution`, `shadowDistance`,
`shadowIntervalSize`, `sunPathRotation`, `*Halflife`, `shadowHardwareFiltering*`, `shadowtexNNearest`,
`voxelDistance`, compute `workGroups` / `workGroupsRender`). Buffer formats and clears are collected over
shadowcomp, begin, prepare, gbuffers, deferred and composite programs in Iris order.

## 4. Properties and id maps

`shaders.properties`, `block.properties`, `item.properties` and `entity.properties` go through the same
preprocessor with the standard macros and option values, then a `java.util.Properties`-style parser.
`PackProperties` understands `image.*` (1D / 2D / 3D / screen-relative, clear flag, sampler name),
`bufferObject.N` (zero-initialised, persistent SSBO at binding N, optionally screen-relative),
`customTexture.*`, `scale.<pass>` (with offsets), `size.buffer.colortexN`, `blend.<program>` and
`blend.<program>.<buffer>`, `flip.<pass>.<buffer>` and `flip.<stage>_pre.*`, `program.<world>/<name>.enabled`,
`uniform.*` / `variable.*`, and flags such as `shadow.enabled`, `separateAo`, `prepareBeforeShadow`,
`shadowTerrain`, `shadowTranslucent`, `shadowEntities`.

**Custom textures**: PNGs (uploaded as stored, RGBA8) or raw files
`path TEXTURE_1D|TEXTURE_2D|TEXTURE_RECTANGLE|TEXTURE_3D internalFormat w [h [d]] pixelFormat pixelType`
with filtering from `<path>.mcmeta` (`blur`, `clamp`; defaults false for PNGs, true for raw textures),
sampled through a sampler object like Iris' `GlSampler` (R wrap always REPEAT).

**Id maps**: `block.<id> = name[:prop=value...]` entries give `mc_Entity.x` for every irisim block (block
states such as `:type=bottom` are matched against irisim's state for that block); `layer.solid|cutout|
cutout_mipped|translucent = <blocks>` moves blocks to another render layer (Iris' render-type override);
`item.properties` gives `heldItemId`, `entity.properties` gives `entityId` of the entity box.

## 5. Programs and schedule

Programs are found per dimension folder (`world0`, `world-1`, `world1`), falling back to the root:
`setup[N].csh`; `shadow.csh` and `shadow_a..z.csh`; `begin|prepare|deferred|composite|shadowcomp[N]`
with `.vsh/.gsh/.fsh` plus `[N].csh` and `[N]_a..z.csh` computes; `final`; every `gbuffers_*` /
`shadow_*` / `dh_*` program, resolved through Iris' `ProgramId` fallback chain. Like Iris'
`ProgramSet.readComputeArray`, a compute array stops at the first missing *or disabled* letter.

### Buffer flipping

Each `colortexN` has a main and an alt texture. irisim replays Iris' `BufferFlipper` over the begin,
prepare, deferred and composite passes before the first frame (`Pipeline._build_schedule`): a pass reads
the side that is current and writes the other one; every buffer it writes flips afterwards, unless
`flip.<pass>.colortexN = false`; `flip.<pass>.colortexN = true` flips without writing; `flip.<group>_pre`
flips before a group. Snapshots of the flip state are taken after `begin` (used by shadow programs),
after `prepare` (opaque gbuffers) and after `deferred` (translucent gbuffers); `final` reads the state after
the last composite. At the end of the frame, buffers left on the alt side that are not cleared every frame
are copied alt to main (Iris' `SwapPass`). `log.txt` prints the schedule
(`RT [0, 3] -> 0A,3M  readsAlt [3]`: writes colortex0 alt and colortex3 main, reads colortex3 from alt).
Shadowcomp passes flip `shadowcolorN` the same way.

### Frame order

1. first frame: clear everything, run `setup*` computes;
2. clear images flagged for clearing, shadow depth / colour clears, `shadow.csh` computes, colour target
   clears (cleared targets: both sides; defaults colortex0 = fog colour, colortex1 = white, others 0);
3. `begin*`;
4. shadow pass (unless `shadow.enabled = false` or no shadow program): solid terrain with `shadow_solid`,
   cutout terrain with `shadow_cutout`, the entity with `shadow_entities`, copy `shadowtex1`, translucent
   terrain with `shadow_water` (each falling back to `shadow`), culling and blending off;
5. `shadowcomp*`, then `prepare*` (`prepareBeforeShadow` swaps the order);
6. sky: `gbuffers_skybasic` (sky colour in `gl_Color`) in the Overworld, `gbuffers_skytextured` in the
   End, none in the Nether; depth writes off;
7. opaque and cutout terrain, the entity (`gbuffers_entities`), optional DH opaque LODs (`dh_terrain`,
   own depth buffer `dhDepthTex0`); copy `depthtex1` / `depthtex2`;
8. `deferred*`;
9. translucent terrain (`gbuffers_water`, back-to-front sorted, default blend
   `SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ONE, ONE_MINUS_SRC_ALPHA` plus `blend.*` overrides), DH water
   (`dh_water`), rain quads (`gbuffers_weather`);
10. `composite*`, `final`, swap.

Compute programs run before the fragment program of the same pass name, with `glMemoryBarrier(ALL)`
before and after each dispatch; `workGroups` / `workGroupsRender` and the queried local size decide the
dispatch size. Per-pass `scale.*` sets the viewport. Mipmaps of `colortexNMipmapEnabled` buffers are
generated before the pass that asks for them.

### Samplers and images

`Pipeline.resolve_sampler` binds each sampler a program declares, by name: `colortexN` (and the legacy
names `gcolor`, `gdepth`, `gnormal`, `composite`, `gaux1-4`) to the side the flip schedule says,
`depthtex0/1/2`, `shadowtex0/1` (with compare mode for `sampler2DShadow` / hardware filtering),
`shadowcolorN`, `noisetex`, `gtexture` / `texture` / `tex` (the block atlas), `normals` / `specular`,
`lightmap`, custom textures and custom images by sampler name, `dhDepthTex0/1`. Images (`colorimgN`,
`shadowcolorimgN`, custom `image.*`) are bound with `glBindImageTexture`. Anything unknown is bound to a
black dummy of the right type and listed at the end of `log.txt`.

## 6. Uniforms

Every frame `cli.Driver.frame_state` computes the standard uniforms: camera position (and its `Int` /
`Fract` split), `gbufferModelView/Projection` and inverses (fov from `--fov`, near 0.05, far = 4 x render
distance; the `far` uniform is the render distance in blocks), previous matrices and positions (identity /
zero on the first frame, like Iris), `sunPosition` / `moonPosition` / `shadowLightPosition` / `upPosition`
from the vanilla celestial angle and `sunPathRotation`, `sunAngle`, `shadowAngle`, the Iris shadow
matrices (light-aligned orthographic ±`shadowDistance`, near -100.05, far 156, `shadowIntervalSize`
snapping), time (`worldTime`, `worldDay`, `moonPhase`, `frameTimeCounter`, `frameTime`, `frameCounter`),
weather (`rainStrength`, `wetness`, `thunderStrength`, `biome_precipitation`, `temperature`, `rainfall`),
`fogColor`, `skyColor`, `eyeBrightness(Smooth)` from the scene light at the eye, `isEyeInWater`,
`centerDepthSmooth` (read back each frame and smoothed with `centerDepthHalflife`), dimension flags,
`heldItemId` / `heldBlockLightValue`, `atlasSize`, and per draw `renderStage`, `entityId`,
`blockEntityId`. With `--dh`: `dhRenderDistance`, `dhNearPlane`, `dhFarPlane`, `dhProjection(Inverse)`,
`dhPreviousProjection`.

**Custom uniforms** (`customuniforms.py`): `uniform.<type>.<name>` and `variable.<type>.<name>` expressions
are translated to Python once and evaluated every frame, in file order, with the standard uniforms (and
earlier variables) as the namespace. Supported: arithmetic, comparisons, `&& || !`, `vec2/3/4`, component
access, `sin cos asin acos tan atan atan2 torad todeg min max clamp abs floor ceil exp exp2 frac log log2 pow
random round signum sqrt fmod if between equals in smooth smoothstep`. `smooth([id,] value, [fadeUp, [fadeDown]])` keeps
its state across frames like OptiFine's. Uniforms a program declares that irisim does not provide are listed in
`log.txt`; they stay at their GLSL default (0).

## 7. The synthetic world

`scene.World` is a 96 x 48 x 96 block grid placed at world (936..1032, 32..80, -1096..-1000), far from the
origin, so precision paths (`cameraPositionInt/Fract`) are exercised: heightmap terrain (grass, dirt, stone,
sand shores), a lake with source blocks at y 62, six oak trees, a stone-brick house with glass windows, a
slab roof, an interior lantern and torch, a stained-glass wall, a hill with a dark tunnel and a chamber
(torch, iron and gold blocks), a lava pool, a glowstone pillar, torches, flowers, grass and an entity box.
The Nether and End variants swap materials (netherrack, soul sand, a lava lake, shroomlight, a ceiling /
end stone, obsidian pillars). Optional test structures (`--glass-gallery`, `--village`, `--gallery`,
`--poles`, `--hall`, `--furnace`, `--test-step`, `--origin-dy`) add test pieces to the Overworld.

Meshing is Sodium-like: faces against opaque neighbours are culled, quads are 4 consecutive vertices
(`gl_VertexID % 4 == 0` starts a quad) drawn through an index buffer, positions are relative to the region
origin. Per vertex: biome tint in `gl_Color.rgb` and vanilla smooth AO (in `gl_Color.a` with
`separateAo = true`, else multiplied into the colour), a smooth lightmap (sky light from column exposure
plus flood fill, block light flood-filled from emitters, vanilla encoding with the vanilla lightmap texture
matrix), normal, `at_tangent` (w = handedness), `mc_midTexCoord`, `mc_Entity` = (block id, render layer),
`at_midBlock` = ((block centre - vertex) x 64, emission). Textures are procedural 16 x 16 sprites in a
256 x 256 atlas with 4 mip levels, packed without padding like the vanilla atlas. `--pbr` adds labPBR
`normals` / `specular` atlases.

## 8. Distant Horizons emulation (`--dh`)

Following Iris' `DHCompatInternal`, `DHTerrainTransformer`, `StandardMacros`, `CommonUniforms` and
`MatrixUniforms`: `DISTANT_HORIZONS` is defined for every program; `dh_terrain` and `dh_water` are compiled
with the DH transformer's inputs (`dhMaterialId`, no fragment `gl_Color`, identity texture matrices,
lightmap `(level + 0.5) / 16`, `gl_ProjectionMatrix` = the DH projection). `scene.build_dh_meshes` builds
an LOD ring of 16-block columns out to `--dh-distance` around the scene (none inside its footprint):
grass, forest canopy, sand, a lake with LOD water, a mountain range, glowing village cells, a lava pool, in
DH-like flat colours. The ring is drawn after the opaque terrain (or before it with `--dh-order before`)
into its own depth buffer `dhDepthTex0`, cleared every frame, back-face culled; its water is drawn after
`gbuffers_water` with `dh_water` (`dhDepthTex1` is a copy taken before it). With `--dh` the default
vanilla render distance drops to 4 chunks so the LODs start right behind the scene.

## 9. Dumps, statistics, timings

`dumps.Dumper` reads back every buffer a pass wrote on dump frames, writes a PNG
(float: Reinhard + sRGB, NaN magenta, +Inf cyan; unorm as is; snorm `*0.5+0.5`; integer: hashed colours;
depth: linearised; 3D images: middle slice, small layer counts as one PNG per layer; integer 3D images as
a top-down height map) and logs `min / max / mean` per channel plus NaN / Inf counts computed on the raw
values. Custom images are dumped after every pass that binds them and at the end of the frame; SSBOs are
written raw. `--npy` and `--npy-seq` save raw arrays instead of PNGs. Timings bracket each pass with
`glFinish` (`--no-timing` removes them) and exclude dump time.

## 10. Differences from Iris worth knowing

See the README's *Limits*. In particular: no particles / hand / block entities / vanilla clouds; culling
is off for translucent terrain and in the shadow pass; memory barriers after every pass; option
discovery scans every shader file under `shaders/` rather than only the include graph of the programs;
conflicting option declarations are resolved by first occurrence instead of being discarded.
