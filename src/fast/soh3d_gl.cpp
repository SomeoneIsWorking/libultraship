// SoH3D direct-GL renderer. See include/fast/soh3d_gl.h.
#ifdef ENABLE_OPENGL

#include "fast/soh3d_gl.h"

// Match the GL headers the OpenGL backend uses (see gfx_opengl.h).
#ifdef _MSC_VER
#include <SDL2/SDL.h>
#include <GL/glew.h>
#elif defined(__APPLE__)
#include <SDL2/SDL.h>
#include <GL/glew.h>
#elif defined(USE_OPENGLES)
#include <SDL2/SDL.h>
#include <GLES3/gl3.h>
#else
#include <SDL2/SDL.h>
#define GL_GLEXT_PROTOTYPES 1
#include <SDL2/SDL_opengl.h>
#endif

#include <vector>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <cstddef>

namespace {

struct GlGroup {
    GLsizei first = 0; // first vertex in the model VBO
    GLsizei count = 0;
    int texIndex = -1;
    int alphaTest = 0;
    float alphaRef = 0;
    GLint wrapS = GL_REPEAT, wrapT = GL_REPEAT;
};

struct GlModel {
    bool uploaded = false;
    bool failed = false;
    GLuint vbo = 0;
    std::vector<GlGroup> groups;
    std::vector<GLuint> textures;
    std::vector<float> bones;  // flat row-major 16*boneCount; empty = bind pose (identity)
    int boneCount = 0;
};

std::unordered_map<int, GlModel> g_models; // keyed by stable model id
SoH3DModelProvider g_provider = nullptr;

GLuint g_program = 0;
GLint g_locPos = -1, g_locNrm = -1, g_locUv = -1, g_locBoneId = -1, g_locBoneW = -1;
GLint g_uMP = -1, g_uInvertY = -1, g_uTint = -1, g_uAlphaRef = -1, g_uTex = -1, g_uBones = -1, g_uSkin = -1;
bool g_progFailed = false;

// GPU skinning: pos_skinned = sum_i aBoneW[i] * uBones[aBoneId[i]] * pos. uBones is
// an array of affine matrices, so the result's w = sum_i aBoneW[i] = 1 (weights sum
// to 1). uBones defaults to identity (set via glUniformMatrix per draw) -> bind pose.
const char* kVert =
    "#version 130\n"
    "in vec3 aPos; in vec3 aNrm; in vec2 aUv; in vec4 aBoneId; in vec4 aBoneW;\n"
    "uniform mat4 uMP; uniform float uInvertY; uniform mat4 uBones[32]; uniform float uSkin;\n"
    "out vec2 vUv;\n"
    "void main(){\n"
    // Skinning (uSkin>0.5) blends the vertex by its bones; at the bind pose / no anim
    // (uSkin==0) this reduces to the raw position (weights sum to 1, uBones identity),
    // so we skip it AND the dynamic uniform-array index uBones[int(aBoneId[i])] — that
    // per-vertex index into a uniform array is undefined-ish on some drivers (ACO on
    // radeonsi collapsed scene geometry to garbage triangles; llvmpipe tolerated it).
    "  vec4 sp;\n"
    "  if (uSkin > 0.5) {\n"
    "    sp = vec4(0.0);\n"
    "    for (int i = 0; i < 4; i++) sp += aBoneW[i] * (uBones[int(aBoneId[i])] * vec4(aPos, 1.0));\n"
    "  } else {\n"
    "    sp = vec4(aPos, 1.0);\n"
    "  }\n"
    "  vec4 c = uMP * vec4(sp.xyz, 1.0);\n"
    "  c.y *= uInvertY;\n"
    "  gl_Position = c;\n"
    "  vUv = vec2(aUv.x, 1.0 - aUv.y);\n" // PICA/CMB UVs are top-origin; GL samples bottom-origin
    "}\n";

const char* kFrag =
    "#version 130\n"
    "in vec2 vUv;\n"
    "uniform sampler2D uTex; uniform vec3 uTint; uniform float uAlphaRef;\n"
    "out vec4 frag;\n"
    "void main(){\n"
    "  vec4 t = texture(uTex, vUv);\n"
    "  if (t.a < uAlphaRef) discard;\n"
    "  frag = vec4(t.rgb * uTint, t.a);\n"
    "}\n";

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "[SoH3D_GL] shader compile failed: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

bool ensureProgram() {
    if (g_program) return true;
    if (g_progFailed) return false;
    GLuint vs = compile(GL_VERTEX_SHADER, kVert);
    GLuint fs = compile(GL_FRAGMENT_SHADER, kFrag);
    if (!vs || !fs) { g_progFailed = true; return false; }
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glBindAttribLocation(p, 0, "aPos");
    glBindAttribLocation(p, 1, "aNrm");
    glBindAttribLocation(p, 2, "aUv");
    glBindAttribLocation(p, 3, "aBoneId");
    glBindAttribLocation(p, 4, "aBoneW");
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "[SoH3D_GL] program link failed: %s\n", log);
        g_progFailed = true;
        return false;
    }
    g_program = p;
    const GLubyte* ver = glGetString(GL_VERSION);
    fprintf(stderr, "[SoH3D_GL] program=%u GL_VERSION=%s\n", g_program, ver ? (const char*)ver : "?");
    g_locPos = 0; g_locNrm = 1; g_locUv = 2; g_locBoneId = 3; g_locBoneW = 4;
    g_uMP = glGetUniformLocation(p, "uMP");
    g_uInvertY = glGetUniformLocation(p, "uInvertY");
    g_uTint = glGetUniformLocation(p, "uTint");
    g_uAlphaRef = glGetUniformLocation(p, "uAlphaRef");
    g_uTex = glGetUniformLocation(p, "uTex");
    g_uBones = glGetUniformLocation(p, "uBones");
    g_uSkin = glGetUniformLocation(p, "uSkin");
    return true;
}

GLint mapWrap(unsigned glWrap) {
    switch (glWrap) {
        case 0x2900: return GL_CLAMP_TO_EDGE; // GL_CLAMP -> clamp to edge
        case 0x812F: return GL_CLAMP_TO_EDGE;
        case 0x8370: return GL_MIRRORED_REPEAT;
        default: return GL_REPEAT; // 0x2901
    }
}

} // namespace

extern "C" void SoH3D_GL_SetModelProvider(SoH3DModelProvider fn) {
    g_provider = fn;
}

extern "C" void SoH3D_GL_SetBones(int modelId, const float* mats16, int n) {
    GlModel& m = g_models[modelId];
    if (n > SOH3D_GL_MAX_BONES) n = SOH3D_GL_MAX_BONES;
    if (!mats16 || n <= 0) { m.bones.clear(); m.boneCount = 0; return; }
    m.bones.assign(mats16, mats16 + (size_t)n * 16);
    m.boneCount = n;
}

// Upload a model's CPU data (from the provider) to GL. GL must be current.
static bool uploadModel(GlModel& m, const SoH3DGlGroup* groups, int groupCount, const SoH3DGlTex* texs, int texCount) {
    std::vector<SoH3DGlVtx> all;
    for (int i = 0; i < groupCount; i++) {
        GlGroup g;
        g.first = (GLsizei)all.size();
        g.count = groups[i].vertCount;
        g.texIndex = groups[i].texIndex;
        g.alphaTest = groups[i].alphaTest;
        g.alphaRef = groups[i].alphaRef;
        g.wrapS = mapWrap(groups[i].wrapS);
        g.wrapT = mapWrap(groups[i].wrapT);
        all.insert(all.end(), groups[i].verts, groups[i].verts + groups[i].vertCount);
        m.groups.push_back(g);
    }
    glGenBuffers(1, &m.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(all.size() * sizeof(SoH3DGlVtx)), all.data(), GL_STATIC_DRAW);
    for (int i = 0; i < texCount; i++) {
        GLuint t = 0;
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texs[i].w, texs[i].h, 0, GL_RGBA, GL_UNSIGNED_BYTE, texs[i].rgba);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        m.textures.push_back(t);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    fprintf(stderr, "[SoH3D_GL] uploaded model: %d groups, %d textures, %zu verts\n", groupCount, texCount, all.size());
    return true;
}

extern "C" void SoH3D_GL_Draw(int modelId, const float* mp16, int invertY, unsigned char r, unsigned char g,
                              unsigned char b, float aspectAdj) {
    if (!ensureProgram()) return;
    GlModel& m = g_models[modelId];
    if (!m.uploaded && !m.failed) {
        const SoH3DGlGroup* groups = nullptr;
        const SoH3DGlTex* texs = nullptr;
        int groupCount = 0, texCount = 0;
        if (g_provider && g_provider(modelId, &groups, &groupCount, &texs, &texCount) && groupCount > 0) {
            uploadModel(m, groups, groupCount, texs, texCount);
            m.uploaded = true;
        } else {
            fprintf(stderr, "[SoH3D_GL] model %d unavailable from provider\n", modelId);
            m.failed = true;
        }
    }
    if (!m.uploaded) return;

    // Diagnostic: SOH3D_GL_NODRAW=1 skips the actual GL draw (keeps the upload) to
    // isolate whether the crash is our draw/state vs the handler/opcode itself.
    static int nodraw = -1;
    if (nodraw < 0) { const char* e = getenv("SOH3D_GL_NODRAW"); nodraw = (e && e[0] == '1') ? 1 : 0; }
    if (nodraw) return;

    // --- save the global GL state we touch. The interpreter renders on the DEFAULT
    // VAO (no VAO on the desktop GL path), so its vertex-attrib pointers live in the
    // global state we're about to overwrite. Save & restore attribs 0..2 exactly
    // (enabled/buffer/size/type/normalized/stride/pointer) so the interpreter's next
    // glDrawArrays still fetches from ITS buffer — not ours (llvmpipe does CPU vertex
    // fetch, so a stale pointer reads past our VBO and segfaults). ---
    GLint prevProg = 0, prevArrayBuf = 0, prevActiveTex = 0, prevTexBind = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProg);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArrayBuf);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
    GLboolean prevBlend = glIsEnabled(GL_BLEND);
    GLboolean prevCull = glIsEnabled(GL_CULL_FACE);
    GLboolean prevDepth = glIsEnabled(GL_DEPTH_TEST);
    GLboolean prevScissor = glIsEnabled(GL_SCISSOR_TEST);
    GLboolean prevDepthMask = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTexBind);

    struct AttribSave {
        GLint enabled, buffer, size, type, normalized, stride;
        void* pointer;
    } as[5];
    for (int i = 0; i < 5; i++) {
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &as[i].enabled);
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &as[i].buffer);
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_SIZE, &as[i].size);
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_TYPE, &as[i].type);
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED, &as[i].normalized);
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &as[i].stride);
        glGetVertexAttribPointerv(i, GL_VERTEX_ATTRIB_ARRAY_POINTER, &as[i].pointer);
    }

    // --- our draw state ---
    glUseProgram(g_program);
    // Mirror Fast3D's per-vertex `x = AdjXForAspectRatio(x)` (interpreter.cpp): scale
    // the clip-space X output of MP by the same factor the N64 actors get. clip.x =
    // sum_k ob[k]*MP[k][0] + MP[3][0], i.e. column 0 of MP in &MP[0][0] row-major =
    // indices 0,4,8,12. Without this the OoT3D scene/models render at the un-squeezed
    // 4:3 X while N64 actors are squeezed to the wide FB -> they shear apart off-center
    // as the camera pans (the "props move differently via camera" bug).
    float mp[16];
    memcpy(mp, mp16, sizeof(mp));
    mp[0] *= aspectAdj;
    mp[4] *= aspectAdj;
    mp[8] *= aspectAdj;
    mp[12] *= aspectAdj;
    glUniformMatrix4fv(g_uMP, 1, GL_FALSE, mp); // row-major matches GLSL col-major load (see header math)
    glUniform1f(g_uInvertY, invertY ? -1.0f : 1.0f);
    glUniform3f(g_uTint, r / 255.0f, g / 255.0f, b / 255.0f);
    glUniform1i(g_uTex, 0);

    // uBones: identity by default (-> bind pose), else the model's per-frame skin
    // matrices. Stored row-major (M*v, like matApplyPos); GLSL does column-major m*v,
    // so upload with transpose=GL_TRUE. Unused slots stay identity.
    {
        float bones[SOH3D_GL_MAX_BONES * 16];
        for (int k = 0; k < SOH3D_GL_MAX_BONES; k++) {
            float* d = bones + k * 16;
            for (int e = 0; e < 16; e++) d[e] = (e % 5 == 0) ? 1.0f : 0.0f; // identity
        }
        int nb = m.boneCount < SOH3D_GL_MAX_BONES ? m.boneCount : SOH3D_GL_MAX_BONES;
        if (!m.bones.empty()) memcpy(bones, m.bones.data(), (size_t)nb * 16 * sizeof(float));
        glUniformMatrix4fv(g_uBones, SOH3D_GL_MAX_BONES, GL_TRUE, bones);
        // Only run the skinning blend (and its per-vertex uniform-array index) when a
        // pose is actually uploaded; rooms / bind-pose models use the raw position.
        glUniform1f(g_uSkin, (!m.bones.empty() && m.boneCount > 0) ? 1.0f : 0.0f);
    }

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
    const GLsizei stride = sizeof(SoH3DGlVtx);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, pos));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, nrm));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, uv));
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, boneIds));
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, weights));

    GLint curFbo = 0, vp[4] = { 0, 0, 0, 0 };
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &curFbo);
    glGetIntegerv(GL_VIEWPORT, vp);
    int totalDrawn = 0;
    for (const GlGroup& grp : m.groups) {
        glUniform1f(g_uAlphaRef, grp.alphaTest ? grp.alphaRef : 0.0f);
        if (grp.texIndex >= 0 && grp.texIndex < (int)m.textures.size()) {
            glBindTexture(GL_TEXTURE_2D, m.textures[grp.texIndex]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, grp.wrapS);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, grp.wrapT);
        }
        glDrawArrays(GL_TRIANGLES, grp.first, grp.count);
        totalDrawn += grp.count;
    }
    {
        static int dbg = -1;
        if (dbg < 0) { const char* e = getenv("SOH3D_GL_DBG"); dbg = (e && e[0] == '1') ? 1 : 0; }
        if (dbg)
            fprintf(stderr,
                    "[SoH3D_GL] drew %d verts -> fbo=%d vp=[%d,%d,%d,%d] invertY=%d glerr=0x%x  MP row0=[%.3f %.3f %.3f "
                    "%.3f] row3=[%.3f %.3f %.3f %.3f]\n",
                    totalDrawn, curFbo, vp[0], vp[1], vp[2], vp[3], invertY, glGetError(), mp16[0], mp16[1], mp16[2],
                    mp16[3], mp16[12], mp16[13], mp16[14], mp16[15]);
    }

    // --- restore Fast3D state, including the exact attrib 0..4 setup (we enable
    // 3/4 for skinning; Fast3D leaves them disabled, so restoring returns them so) ---
    for (int i = 0; i < 5; i++) {
        glBindBuffer(GL_ARRAY_BUFFER, (GLuint)as[i].buffer);
        glVertexAttribPointer(i, as[i].size, (GLenum)as[i].type, (GLboolean)as[i].normalized, as[i].stride,
                              as[i].pointer);
        if (as[i].enabled) glEnableVertexAttribArray(i); else glDisableVertexAttribArray(i);
    }
    glBindTexture(GL_TEXTURE_2D, (GLuint)prevTexBind);
    glActiveTexture((GLenum)prevActiveTex);
    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)prevArrayBuf);
    glUseProgram((GLuint)prevProg);
    if (prevBlend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
    if (prevCull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    if (prevDepth) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
    if (prevScissor) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
    glDepthMask(prevDepthMask);
}

#endif // ENABLE_OPENGL
