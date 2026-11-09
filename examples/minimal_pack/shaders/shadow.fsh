#version 330 compatibility

uniform sampler2D gtexture;

in vec2 texcoord;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 color;

void main() {
    vec4 albedo = texture(gtexture, texcoord);
    if (albedo.a < 0.1) discard;
    color = albedo;
}
