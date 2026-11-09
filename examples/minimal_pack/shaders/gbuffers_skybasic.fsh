#version 330 compatibility

in vec4 glcolor;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 color;

void main() {
    color = vec4(glcolor.rgb, 1.0);
}
