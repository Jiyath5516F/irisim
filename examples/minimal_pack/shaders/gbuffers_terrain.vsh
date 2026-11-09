#version 330 compatibility
#include "/lib/settings.glsl"

uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
in vec4 mc_Entity;

out vec2 texcoord;
out vec2 lmcoord;
out vec4 glcolor;
out vec3 normal;
out vec3 shadowPos;
flat out int blockId;

void main() {
    vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    gl_Position = gl_ProjectionMatrix * viewPos;
    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    glcolor = gl_Color;
    normal = normalize(gl_NormalMatrix * gl_Normal);
    vec4 playerPos = gbufferModelViewInverse * viewPos;
    vec4 sp = shadowProjection * (shadowModelView * playerPos);
    shadowPos = sp.xyz / sp.w * 0.5 + 0.5;
    blockId = int(mc_Entity.x + 0.5);
}
