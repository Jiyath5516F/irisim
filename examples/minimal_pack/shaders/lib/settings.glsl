#ifndef SETTINGS_GLSL
#define SETTINGS_GLSL

#define SHADOWS                 // Sun shadows from shadowtex0
#define VIGNETTE                // Darken the screen corners
//#define DEBUG_NORMALS         // Show gbuffer normals instead of the image
#define FOG_DENSITY 1.0         // [0.0 0.5 1.0 2.0]
#define EXPOSURE 1.0            // [0.5 1.0 1.5 2.0]

const int shadowMapResolution = 1024; // [512 1024 2048]
const float shadowDistance = 64.0;    // [32.0 64.0 128.0]
const float sunPathRotation = -30.0;  // [-45.0 -30.0 0.0 30.0]

#endif
