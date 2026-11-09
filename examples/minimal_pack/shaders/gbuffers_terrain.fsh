#version 330 compatibility
#include "/lib/settings.glsl"

uniform sampler2D gtexture;
uniform sampler2D lightmap;
uniform sampler2D shadowtex0;
uniform vec3 shadowLightPosition;

in vec2 texcoord;
in vec2 lmcoord;
in vec4 glcolor;
in vec3 normal;
in vec3 shadowPos;
flat in int blockId;

/* RENDERTARGETS: 0,2 */
layout(location = 0) out vec4 color;
layout(location = 1) out vec4 normalOut;

void main() {
    vec4 albedo = texture(gtexture, texcoord) * glcolor;
    if (albedo.a < 0.1) discard;
    float ndl = max(dot(normal, normalize(shadowLightPosition)), 0.0);
    float lit = 1.0;
#ifdef SHADOWS
    if (all(greaterThan(shadowPos, vec3(0.0))) && all(lessThan(shadowPos, vec3(1.0)))) {
        lit = step(shadowPos.z - 0.001, texture(shadowtex0, shadowPos.xy).r);
    }
#endif
    vec3 ambient = texture(lightmap, lmcoord).rgb * 0.45;
    vec3 sun = vec3(1.0, 0.95, 0.85) * ndl * lit * lmcoord.y;
    if (blockId == 20) albedo.a = 0.7;
    color = vec4(albedo.rgb * (ambient + sun), albedo.a);
    normalOut = vec4(normal * 0.5 + 0.5, 1.0);
}
