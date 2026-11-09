#version 330 compatibility
#include "/lib/settings.glsl"

uniform sampler2D colortex0;
uniform sampler2D colortex2;
uniform sampler2D depthtex0;
uniform float near;
uniform float far;
uniform vec3 fogColor;

in vec2 texcoord;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 color;

float linearDepth(float d) {
    return 2.0 * near * far / (far + near - (d * 2.0 - 1.0) * (far - near));
}

void main() {
    vec3 c = texture(colortex0, texcoord).rgb;
    float depth = texture(depthtex0, texcoord).r;
    if (depth < 1.0) {
        float dist = linearDepth(depth);
        float fog = 1.0 - exp(-dist / far * FOG_DENSITY * 2.0);
        c = mix(c, fogColor, clamp(fog, 0.0, 1.0));
    }
#ifdef DEBUG_NORMALS
    c = texture(colortex2, texcoord).rgb;
#endif
    color = vec4(c, 1.0);
}
