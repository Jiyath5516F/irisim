#version 330 compatibility
#include "/lib/settings.glsl"

uniform sampler2D colortex0;

in vec2 texcoord;

layout(location = 0) out vec4 color;

void main() {
    vec3 c = texture(colortex0, texcoord).rgb * EXPOSURE;
    c = c / (1.0 + c) * 1.6;
#ifdef VIGNETTE
    vec2 d = texcoord - 0.5;
    c *= 1.0 - dot(d, d) * 0.8;
#endif
    color = vec4(c, 1.0);
}
