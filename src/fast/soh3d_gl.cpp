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
    int blendEnable = 0;
    GLenum blendSrcRGB = GL_SRC_ALPHA, blendDstRGB = GL_ONE_MINUS_SRC_ALPHA, blendEqRGB = GL_FUNC_ADD;
    GLenum blendSrcA = GL_ONE, blendDstA = GL_ZERO, blendEqA = GL_FUNC_ADD;
    float blendColor[4] = { 0, 0, 0, 1 };
    int depthWrite = 1;
    float polygonOffset = 0.0f; // window-depth bias for decals (gl_FragDepth += this)
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

// Emit-time pose snapshots, captured when each actor's draw opcode is written (after its SetBones,
// before later same-modelId actors overwrite g_models[modelId].bones). Stored per modelId in EMIT
// ORDER for THIS logic frame (g_curPoses) and the previous one (g_prevPoses). The k-th submit of a
// modelId in a subframe pairs with the k-th emit, so same-model actors keep their own poses AND we
// can interpolate each between its previous- and current-frame pose (see SoH3D_GL_RenderPass).
struct ItemPose {
    std::vector<float> bones;
    int boneCount = 0;
};
std::unordered_map<int, std::vector<ItemPose>> g_curPoses;  // this logic frame, per modelId
std::unordered_map<int, std::vector<ItemPose>> g_prevPoses; // last logic frame, per modelId

// Frame-interpolation step for the CURRENT subframe replay (0 = previous logic frame, 1 = current),
// set per subframe by RunCommands (OTRGlobals.cpp). The game records gfx once per logic frame and
// replays it N times with interpolated matrices; we lerp each bone pose by this so the skinned
// limbs interpolate to the render FPS like the N64 matrix stack does, instead of snapping at 20fps.
extern "C" float gSoH3dInterpStep = 1.0f;

GLuint g_program = 0;
// Our own Vertex Array Object. Fast3D's GL backend renders on its OWN VAO (gfx_opengl.cpp
// creates mOpenglVao once at init and assumes it stays configured), so we must NOT mutate
// the bound VAO's attrib state — we draw inside g_vao and restore the previous VAO binding,
// leaving Fast3D's vertex state pristine. This is what prevents our attrib setup leaking into
// Fast3D's 2D/skybox draws (the recurring striped-UI corruption). See [[soh3d-gl-state-leak]].
GLuint g_vao = 0;
GLint g_locPos = -1, g_locNrm = -1, g_locUv = -1, g_locBoneId = -1, g_locBoneW = -1;
GLint g_uMP = -1, g_uInvertY = -1, g_uTint = -1, g_uAlphaRef = -1, g_uTex = -1, g_uBones = -1, g_uSkin = -1;
GLint g_uDepthOffset = -1, g_uMV = -1, g_uLit = -1, g_uLightDir = -1;
bool g_progFailed = false;

// GPU skinning: pos_skinned = sum_i aBoneW[i] * uBones[aBoneId[i]] * pos. uBones is
// an array of affine matrices, so the result's w = sum_i aBoneW[i] = 1 (weights sum
// to 1). uBones defaults to identity (set via glUniformMatrix per draw) -> bind pose.
const char* kVert =
    "#version 130\n"
    "in vec3 aPos; in vec3 aNrm; in vec2 aUv; in vec4 aBoneId; in vec4 aBoneW; in vec4 aColor;\n"
    "uniform mat4 uMP; uniform mat4 uMV; uniform float uInvertY; uniform mat4 uBones[32]; uniform float uSkin;\n"
    "out vec2 vUv; out vec4 vColor; out vec3 vNrmView;\n"
    "void main(){\n"
    "  vColor = aColor;\n"
    // Skinning (uSkin>0.5) blends the vertex by its bones; at the bind pose / no anim
    // (uSkin==0) this reduces to the raw position (weights sum to 1, uBones identity),
    // so we skip it AND the dynamic uniform-array index uBones[int(aBoneId[i])] — that
    // per-vertex index into a uniform array is undefined-ish on some drivers (ACO on
    // radeonsi collapsed scene geometry to garbage triangles; llvmpipe tolerated it).
    "  vec4 sp;\n"
    "  vec3 nM;\n"
    "  if (uSkin > 0.5) {\n"
    "    sp = vec4(0.0); nM = vec3(0.0);\n"
    "    for (int i = 0; i < 4; i++) {\n"
    "      sp += aBoneW[i] * (uBones[int(aBoneId[i])] * vec4(aPos, 1.0));\n"
    "      nM += aBoneW[i] * (mat3(uBones[int(aBoneId[i])]) * aNrm);\n" // skin the normal too (for lighting)
    "    }\n"
    "  } else {\n"
    "    sp = vec4(aPos, 1.0); nM = aNrm;\n"
    "  }\n"
    "  vec4 c = uMP * vec4(sp.xyz, 1.0);\n"
    "  c.y *= uInvertY;\n"
    "  gl_Position = c;\n"
    // WORLD-space normal for the fragment lighting term. uMV is the N64 "modelview" = the
    // model->world matrix ONLY: OoT folds the camera/viewing transform into the PROJECTION
    // matrix (z_view.c loads viewing with G_MTX_PROJECTION|G_MTX_MUL), so the modelview stack
    // top carries no view. Hence mat3(uMV)*nM lands in WORLD space, and the light dir we
    // compare against (uLightDir) is the scene's world-space sun direction. Uniform model
    // scale -> mat3(uMV) is rotation*scale; the frag renormalizes.
    "  vNrmView = mat3(uMV) * nM;\n"
    "  vUv = vec2(aUv.x, 1.0 - aUv.y);\n" // PICA/CMB UVs are top-origin; GL samples bottom-origin
    "}\n";

const char* kFrag =
    "#version 130\n"
    "in vec2 vUv; in vec4 vColor; in vec3 vNrmView;\n"
    "uniform sampler2D uTex; uniform vec3 uTint; uniform float uAlphaRef;\n"
    "uniform float uDepthOffset; uniform float uLit; uniform vec3 uLightDir;\n"
    "out vec4 frag;\n"
    // uLightDir = the scene's WORLD-space key-light (sun) direction TO the light, set per frame
    // from play->envCtx.lightSettings.light1Dir (soh3d.c SoH3D_UpdateLight) so the form shading
    // tracks time of day / the world, not the camera. The scene's colour still comes from uTint;
    // this only shapes brightness across the surface. (vNrmView is a world-space normal, see vert.)
    "void main(){\n"
    "  vec4 t = texture(uTex, vUv);\n"
    "  if (t.a < uAlphaRef) discard;\n"
    // Decal depth bias (OoT3D polygon offset): pull flagged coplanar decals toward the
    // camera so they don't z-fight the base ground/wall. 0 for normal materials.
    "  gl_FragDepth = gl_FragCoord.z + uDepthOffset;\n"
    // OoT3D modulates the texture by the per-vertex color (baked scene lighting: dimmed
    // walls, ground AO) and the vertex alpha, then by the scene-ambient tint. Character/prop
    // models carry NO baked lighting (flat vColor) -> they looked flat; add a half-Lambert
    // diffuse FORM term (uLit) using the model normal so they read as 3D. Scene geometry
    // (uLit==0) keeps its baked vColor untouched.
    "  vec3 shade = uTint;\n"
    "  if (uLit > 0.5) {\n"
    "    float hl = dot(normalize(vNrmView), normalize(uLightDir)) * 0.5 + 0.5;\n" // half-Lambert wrap [0,1]
    "    shade = uTint * (0.55 + 0.45 * hl);\n"                          // 0.55 ambient floor -> never black
    "  }\n"
    "  frag = vec4(t.rgb * vColor.rgb * shade, t.a * vColor.a);\n"
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
    glBindAttribLocation(p, 5, "aColor");
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
    g_uDepthOffset = glGetUniformLocation(p, "uDepthOffset");
    g_uMV = glGetUniformLocation(p, "uMV");
    g_uLit = glGetUniformLocation(p, "uLit");
    g_uLightDir = glGetUniformLocation(p, "uLightDir");
    glGenVertexArrays(1, &g_vao); // our isolated VAO (never touch Fast3D's)
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

// Character/prop lighting gate, toggled by soh3d.c's REPL (`light 0|1`) and seeded from env
// SOH3D_LIGHT. -1 = uninit (read env on first draw), 0 = off (flat tint), 1 = on (half-Lambert form).
extern "C" int gSoH3dLightEnable = -1;

// World-space key-light (sun) direction TO the light, set once per frame by soh3d.c
// (SoH3D_UpdateLight, from envCtx.lightSettings.light1Dir) and read by the render pass into
// uLightDir. Default = the old fixed direction so legacy/uninit draws look as before.
extern "C" float gSoH3dLightDirWorld[3] = { 0.40f, 0.55f, 0.73f };

extern "C" void SoH3D_GL_SetLightDir(const float dirWorld[3]) {
    gSoH3dLightDirWorld[0] = dirWorld[0];
    gSoH3dLightDirWorld[1] = dirWorld[1];
    gSoH3dLightDirWorld[2] = dirWorld[2];
}

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

extern "C" void SoH3D_GL_EmitPose(int modelId) {
    // Snapshot this actor's just-set pose at EMIT time (during dlist build, logic-frame rate) so it
    // survives later same-modelId SetBones calls. Appended in emit order; the k-th submit of this
    // modelId in a subframe pairs with the k-th entry here. Called from SoH3D_EmitModelDraw before
    // the draw opcode. No bones set -> push an empty entry so emit/submit stay 1:1.
    ItemPose p;
    auto it = g_models.find(modelId);
    if (it != g_models.end() && !it->second.bones.empty()) {
        p.bones = it->second.bones;
        p.boneCount = it->second.boneCount;
    }
    g_curPoses[modelId].push_back(std::move(p));
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
        g.blendEnable = groups[i].blendEnable;
        g.blendSrcRGB = groups[i].blendSrcRGB;
        g.blendDstRGB = groups[i].blendDstRGB;
        g.blendEqRGB = groups[i].blendEqRGB;
        g.blendSrcA = groups[i].blendSrcA;
        g.blendDstA = groups[i].blendDstA;
        g.blendEqA = groups[i].blendEqA;
        g.depthWrite = groups[i].depthWrite;
        g.polygonOffset = groups[i].polygonOffset;
        for (int k = 0; k < 4; k++) g.blendColor[k] = groups[i].blendColor[k];
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

// Ensure a model's GPU data is uploaded (lazy, via the provider). Returns the model or
// nullptr if it has no usable geometry. GL must be current.
static GlModel* ensureUploaded(int modelId) {
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
    return m.uploaded ? &m : nullptr;
}

namespace {

// Fast3D GL state we touch and must hand back exactly as it was (or as gfx_opengl assumes
// it constant). All vertex-array state is isolated in g_vao, so only this global context
// state needs explicit save/restore. See [[soh3d-gl-state-leak]] / gfx_opengl.cpp.
struct SavedGl {
    GLint vao, prog, arrayBuf, activeTex, texBind, depthFunc;
    GLboolean blend, cull, depth, scissor, depthMask;
};

// --- GL state-leak DETECTOR (env SOH3D_GL_STATECHECK=1) ------------------------------------------
// The non-deterministic skybox/HUD stripe corruption is consistent with our render pass leaving
// some global GL state un-restored; the NEXT frame's Fast3D skybox (drawn BEFORE our pass) then
// inherits it, and what leaks depends on which groups/blend states we drew last -> non-deterministic.
// This snapshots a BROAD set of context + bound-VAO attrib state; RenderPass compares pre-pass vs
// post-pass and logs any field we failed to hand back. A CLEAN diff rules a state leak OUT.
struct FullGl {
    GLint vao, prog, arrBuf, elemBuf, activeTex, tex0, tex1, depthFunc, cullMode, frontFace;
    GLint bSrcRGB, bDstRGB, bSrcA, bDstA, bEqRGB, bEqA, viewport[4];
    GLfloat blendColor[4], depthRange[2];
    GLboolean blend, cull, depth, scissor, depthMask, colorMask[4];
    GLint attrEn[8], attrBuf[8];
};
static void captureFullGl(FullGl& f) {
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &f.vao);
    glGetIntegerv(GL_CURRENT_PROGRAM, &f.prog);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &f.arrBuf);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &f.elemBuf);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &f.activeTex);
    glActiveTexture(GL_TEXTURE0); glGetIntegerv(GL_TEXTURE_BINDING_2D, &f.tex0);
    glActiveTexture(GL_TEXTURE1); glGetIntegerv(GL_TEXTURE_BINDING_2D, &f.tex1);
    glActiveTexture((GLenum)f.activeTex);
    glGetIntegerv(GL_BLEND_SRC_RGB, &f.bSrcRGB); glGetIntegerv(GL_BLEND_DST_RGB, &f.bDstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &f.bSrcA); glGetIntegerv(GL_BLEND_DST_ALPHA, &f.bDstA);
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &f.bEqRGB); glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &f.bEqA);
    glGetIntegerv(GL_DEPTH_FUNC, &f.depthFunc); glGetIntegerv(GL_CULL_FACE_MODE, &f.cullMode);
    glGetIntegerv(GL_FRONT_FACE, &f.frontFace); glGetIntegerv(GL_VIEWPORT, f.viewport);
    glGetFloatv(GL_BLEND_COLOR, f.blendColor); glGetFloatv(GL_DEPTH_RANGE, f.depthRange);
    f.blend = glIsEnabled(GL_BLEND); f.cull = glIsEnabled(GL_CULL_FACE);
    f.depth = glIsEnabled(GL_DEPTH_TEST); f.scissor = glIsEnabled(GL_SCISSOR_TEST);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &f.depthMask); glGetBooleanv(GL_COLOR_WRITEMASK, f.colorMask);
    for (int i = 0; i < 8; i++) {
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &f.attrEn[i]);
        glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &f.attrBuf[i]);
    }
}
// -1 = uninit (seed from env SOH3D_GL_STATECHECK on first use), 0 = off, 1 = on. Cross-module so
// soh3d.c's REPL `statecheck 1` can flip it on the moment corruption appears (no relaunch).
extern "C" int gSoH3dStateCheck = -1;
static void checkGlLeak(const FullGl& pre, const char* where) {
    if (gSoH3dStateCheck < 0) { const char* e = getenv("SOH3D_GL_STATECHECK"); gSoH3dStateCheck = (e && e[0] == '1') ? 1 : 0; }
    if (!gSoH3dStateCheck) return;
    FullGl p; captureFullGl(p);
    int n = 0;
#define LK_I(field) if (pre.field != p.field) { fprintf(stderr, "[SoH3D_GL LEAK %s] %s: %d -> %d\n", where, #field, (int)pre.field, (int)p.field); n++; }
    LK_I(vao) LK_I(prog) LK_I(arrBuf) LK_I(elemBuf) LK_I(activeTex) LK_I(tex0) LK_I(tex1)
    LK_I(depthFunc) LK_I(cullMode) LK_I(frontFace) LK_I(bSrcRGB) LK_I(bDstRGB) LK_I(bSrcA) LK_I(bDstA)
    LK_I(bEqRGB) LK_I(bEqA) LK_I(blend) LK_I(cull) LK_I(depth) LK_I(scissor) LK_I(depthMask)
#undef LK_I
    if (memcmp(pre.viewport, p.viewport, sizeof(p.viewport))) { fprintf(stderr, "[SoH3D_GL LEAK %s] viewport changed\n", where); n++; }
    if (memcmp(pre.blendColor, p.blendColor, sizeof(p.blendColor))) { fprintf(stderr, "[SoH3D_GL LEAK %s] blendColor changed\n", where); n++; }
    if (memcmp(pre.depthRange, p.depthRange, sizeof(p.depthRange))) { fprintf(stderr, "[SoH3D_GL LEAK %s] depthRange changed\n", where); n++; }
    if (memcmp(pre.colorMask, p.colorMask, sizeof(p.colorMask))) { fprintf(stderr, "[SoH3D_GL LEAK %s] colorMask changed\n", where); n++; }
    for (int i = 0; i < 8; i++) {
        if (pre.attrEn[i] != p.attrEn[i]) { fprintf(stderr, "[SoH3D_GL LEAK %s] attrib[%d] enabled %d -> %d\n", where, i, pre.attrEn[i], p.attrEn[i]); n++; }
        if (pre.attrBuf[i] != p.attrBuf[i]) { fprintf(stderr, "[SoH3D_GL LEAK %s] attrib[%d] buffer %d -> %d\n", where, i, pre.attrBuf[i], p.attrBuf[i]); n++; }
    }
    if (n) fprintf(stderr, "[SoH3D_GL LEAK %s] %d field(s) NOT restored by our pass\n", where, n);
}

// Open our render pass: snapshot Fast3D's state, then install OUR common state once (isolated
// VAO, our program, depth test on / LEQUAL, scissor+cull off). Per-item uniforms/attribs and
// per-group blend/depth-write are set inside drawOne.
void beginPass(SavedGl& s) {
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &s.vao);
    glGetIntegerv(GL_CURRENT_PROGRAM, &s.prog);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &s.arrayBuf);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &s.activeTex);
    glGetIntegerv(GL_DEPTH_FUNC, &s.depthFunc);
    s.blend = glIsEnabled(GL_BLEND);
    s.cull = glIsEnabled(GL_CULL_FACE);
    s.depth = glIsEnabled(GL_DEPTH_TEST);
    s.scissor = glIsEnabled(GL_SCISSOR_TEST);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &s.depthMask);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &s.texBind);

    glBindVertexArray(g_vao); // our isolated VAO; attrib changes stay here, off Fast3D's VAO
    glUseProgram(g_program);
    glUniform1i(g_uTex, 0);
    glUniform3fv(g_uLightDir, 1, gSoH3dLightDirWorld); // scene sun dir (world space), per frame
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);
    // Blend + depth-write are set per group from the CMB material inside drawOne.
}

// Close our render pass: restore everything to Fast3D's snapshot, and deterministically reset
// the state gfx_opengl sets ONCE at init and never again (blendFunc/equation, depthFunc) so its
// implicit cache stays consistent with GL. NOT a glGet round-trip for those (see memory: that
// restored garbage). depthFunc IS save/restored because it's the live value Fast3D last set.
void endPass(const SavedGl& s) {
    glBindVertexArray((GLuint)s.vao); // restores ALL Fast3D vertex-array state in one shot
    glBindTexture(GL_TEXTURE_2D, (GLuint)s.texBind);
    glActiveTexture((GLenum)s.activeTex);
    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)s.arrayBuf);
    glUseProgram((GLuint)s.prog);
    if (s.blend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
    if (s.cull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    if (s.depth) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
    if (s.scissor) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
    glDepthMask(s.depthMask);
    glDepthFunc((GLenum)s.depthFunc);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // gfx_opengl's permanent init assumption
    glBlendEquation(GL_FUNC_ADD);
    glBlendColor(0.0f, 0.0f, 0.0f, 0.0f); // we set blendColor per group; Fast3D never does -> reset
}

// Draw one already-uploaded model with the given MP/invertY/tint, using the model's currently
// set skinning pose. Assumes beginPass installed the common state and the VAO is g_vao.
void drawOne(GlModel& m, const float* mp16, const float* mv16, int lit, int invertY, unsigned char r,
             unsigned char g, unsigned char b, float aspectAdj, const float* boneData, int boneCnt) {
    // Mirror Fast3D's per-vertex `x = AdjXForAspectRatio(x)` (interpreter.cpp): scale the
    // clip-space X output of MP by the factor the N64 actors get (MP column 0 = row-major
    // indices 0,4,8,12). Without it the OoT3D content shears vs N64 actors as the camera pans.
    float mp[16];
    memcpy(mp, mp16, sizeof(mp));
    mp[0] *= aspectAdj;
    mp[4] *= aspectAdj;
    mp[8] *= aspectAdj;
    mp[12] *= aspectAdj;
    glUniformMatrix4fv(g_uMP, 1, GL_FALSE, mp); // row-major matches GLSL col-major load (header math)
    // Modelview (no projection, no aspect squeeze) -> view-space normal for the lighting term.
    // Global gate (REPL `light 0|1` / env SOH3D_LIGHT, default on) to A/B or disable the form term.
    if (gSoH3dLightEnable < 0) { const char* e = getenv("SOH3D_LIGHT"); gSoH3dLightEnable = (e && e[0] == '0') ? 0 : 1; }
    glUniformMatrix4fv(g_uMV, 1, GL_FALSE, mv16);
    glUniform1f(g_uLit, (lit && gSoH3dLightEnable) ? 1.0f : 0.0f);
    glUniform1f(g_uInvertY, invertY ? -1.0f : 1.0f);
    glUniform3f(g_uTint, r / 255.0f, g / 255.0f, b / 255.0f);

    // uBones: identity by default (bind pose), else THIS draw item's per-frame skin matrices
    // (boneData/boneCnt, snapshotted at Submit time so two actors sharing a modelId keep their own
    // poses). Row-major (M*v), uploaded transposed for GLSL's column-major m*v. Unused slots identity.
    {
        float bones[SOH3D_GL_MAX_BONES * 16];
        for (int k = 0; k < SOH3D_GL_MAX_BONES; k++) {
            float* d = bones + k * 16;
            for (int e = 0; e < 16; e++) d[e] = (e % 5 == 0) ? 1.0f : 0.0f; // identity
        }
        int nb = boneCnt < SOH3D_GL_MAX_BONES ? boneCnt : SOH3D_GL_MAX_BONES;
        if (boneData && boneCnt > 0) memcpy(bones, boneData, (size_t)nb * 16 * sizeof(float));
        glUniformMatrix4fv(g_uBones, SOH3D_GL_MAX_BONES, GL_TRUE, bones);
        glUniform1f(g_uSkin, (boneData && boneCnt > 0) ? 1.0f : 0.0f);
    }

    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    for (int a = 0; a <= 5; a++) glEnableVertexAttribArray(a);
    const GLsizei stride = sizeof(SoH3DGlVtx);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, pos));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, nrm));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, uv));
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, boneIds));
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, weights));
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(SoH3DGlVtx, color));

    for (const GlGroup& grp : m.groups) {
        glUniform1f(g_uAlphaRef, grp.alphaTest ? grp.alphaRef : 0.0f);
        glUniform1f(g_uDepthOffset, grp.polygonOffset);
        if (grp.blendEnable) {
            glEnable(GL_BLEND);
            glBlendFuncSeparate(grp.blendSrcRGB, grp.blendDstRGB, grp.blendSrcA, grp.blendDstA);
            glBlendEquationSeparate(grp.blendEqRGB, grp.blendEqA);
            glBlendColor(grp.blendColor[0], grp.blendColor[1], grp.blendColor[2], grp.blendColor[3]);
        } else {
            glDisable(GL_BLEND);
        }
        glDepthMask(grp.depthWrite ? GL_TRUE : GL_FALSE);
        if (grp.texIndex >= 0 && grp.texIndex < (int)m.textures.size()) {
            glBindTexture(GL_TEXTURE_2D, m.textures[grp.texIndex]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, grp.wrapS);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, grp.wrapT);
        }
        glDrawArrays(GL_TRIANGLES, grp.first, grp.count);
    }
}

// One collected draw (captured at OTR_G_SOH3D_DRAW time; rendered later in the pass).
struct DrawItem {
    int modelId;
    float mp[16];
    float mv[16]; // modelview (for the view-space normal lighting term)
    int lit;      // 1 = apply the half-Lambert form term (characters/props); 0 = scene geometry
    int invertY;
    unsigned char r, g, b;
    float aspectAdj;
    std::vector<float> bones;     // this-frame skin pose (so same-modelId actors keep own poses)
    std::vector<float> prevBones; // same item's previous-frame pose (for FPS interpolation); may be empty
    int boneCount = 0;
};
std::vector<DrawItem> g_drawList;

} // namespace

// Inline single-model draw (legacy entry; still used by any direct caller). Brackets one model
// in its own pass. The collected path (Submit/RenderPass) is preferred — it brackets the whole
// frame's SoH3D content once.
extern "C" void SoH3D_GL_Draw(int modelId, const float* mp16, int invertY, unsigned char r, unsigned char g,
                              unsigned char b, float aspectAdj) {
    if (!ensureProgram()) return;
    GlModel* m = ensureUploaded(modelId);
    if (!m) return;
    static int nodraw = -1;
    if (nodraw < 0) { const char* e = getenv("SOH3D_GL_NODRAW"); nodraw = (e && e[0] == '1') ? 1 : 0; }
    if (nodraw) return;
    SavedGl s;
    beginPass(s);
    // legacy path: no lighting; pose = the model's current bones (single-actor inline draw)
    drawOne(*m, mp16, mp16, /*lit=*/0, invertY, r, g, b, aspectAdj,
            m->bones.empty() ? nullptr : m->bones.data(), m->boneCount);
    endPass(s);
}

extern "C" void SoH3D_GL_Submit(int modelId, const float* mp16, const float* mv16, int lit, int invertY,
                                unsigned char r, unsigned char g, unsigned char b, float aspectAdj) {
    DrawItem it;
    it.modelId = modelId;
    memcpy(it.mp, mp16, sizeof(it.mp));
    memcpy(it.mv, mv16 ? mv16 : mp16, sizeof(it.mv));
    it.lit = lit;
    it.invertY = invertY;
    it.r = r;
    it.g = g;
    it.b = b;
    it.aspectAdj = aspectAdj;
    // Per-item pose pairing. Submit runs at dlist INTERPRET time (and re-runs once per interpolation
    // subframe), by which point g_models[modelId].bones holds only the LAST actor's pose. So pair by
    // EMIT ORDER: this is the k-th submit of `modelId` in the current subframe (k = how many items of
    // this modelId are already collected), which corresponds to the k-th EmitPose this logic frame.
    // Carry both that pose (cur) and the same slot's previous-frame pose (prev) for FPS interpolation.
    size_t k = 0;
    for (const DrawItem& d : g_drawList)
        if (d.modelId == modelId) k++;
    auto cit = g_curPoses.find(modelId);
    if (cit != g_curPoses.end() && k < cit->second.size() && !cit->second[k].bones.empty()) {
        it.bones = cit->second[k].bones;
        it.boneCount = cit->second[k].boneCount;
        auto pit = g_prevPoses.find(modelId);
        if (pit != g_prevPoses.end() && k < pit->second.size() && pit->second[k].boneCount == it.boneCount)
            it.prevBones = pit->second[k].bones; // same skeleton last frame -> interpolate toward cur
    } else {
        // No emit-time pose (legacy inline path / unposed model): fall back to the model's bones.
        auto mit = g_models.find(modelId);
        if (mit != g_models.end() && !mit->second.bones.empty()) {
            it.bones = mit->second.bones;
            it.boneCount = mit->second.boneCount;
        }
    }
    g_drawList.push_back(std::move(it));
}

extern "C" void SoH3D_GL_FrameBegin(void) {
    g_drawList.clear();
    // Rotate this logic frame's emit-ordered poses into "previous" so the next frame can interpolate
    // each item from where it was. (Called once per logic frame, before the actors emit their poses.)
    g_prevPoses = std::move(g_curPoses);
    g_curPoses.clear();
}

extern "C" void SoH3D_GL_RenderPass(void) {
    if (g_drawList.empty()) return;
    if (!ensureProgram()) { g_drawList.clear(); return; }
    static int nodraw = -1;
    if (nodraw < 0) { const char* e = getenv("SOH3D_GL_NODRAW"); nodraw = (e && e[0] == '1') ? 1 : 0; }
    if (nodraw) { g_drawList.clear(); return; }

    FullGl pre;
    if (gSoH3dStateCheck != 0) captureFullGl(pre); // snapshot BEFORE the pass (skipped once STATECHECK confirmed off)
    SavedGl s;
    beginPass(s);
    int drawn = 0;
    // Interpolate each item's skin pose toward this subframe's step, matching the per-subframe matrix
    // interpolation the rest of the scene gets — so skinned limbs animate at the render FPS instead of
    // snapping at the 20fps logic rate. Component-wise matrix lerp, the same blend frame_interpolation
    // applies to recorded N64 matrices. step>=1 or no prev pose -> use cur directly (no work).
    std::vector<float> lerped;
    float step = gSoH3dInterpStep;
    for (const DrawItem& it : g_drawList) {
        GlModel* m = ensureUploaded(it.modelId);
        if (!m) continue;
        const float* pose = it.bones.empty() ? nullptr : it.bones.data();
        if (pose && step < 0.999f && !it.prevBones.empty() && it.prevBones.size() == it.bones.size()) {
            lerped.resize(it.bones.size());
            float w = 1.0f - step;
            for (size_t i = 0; i < it.bones.size(); i++) lerped[i] = w * it.prevBones[i] + step * it.bones[i];
            pose = lerped.data();
        }
        drawOne(*m, it.mp, it.mv, it.lit, it.invertY, it.r, it.g, it.b, it.aspectAdj, pose, it.boneCount);
        drawn++;
    }
    endPass(s);
    checkGlLeak(pre, "renderpass"); // verify our pass handed every captured state field back

    {
        static int dbg = -1;
        if (dbg < 0) { const char* e = getenv("SOH3D_GL_DBG"); dbg = (e && e[0] == '1') ? 1 : 0; }
        if (dbg) {
            fprintf(stderr, "[SoH3D_GL] render pass: %d/%zu items glerr=0x%x\n", drawn, g_drawList.size(),
                    glGetError());
            // Per-item pose checksum: two items with the same modelId but DIFFERENT sums prove the
            // per-item pose capture works (the old per-modelId store gave same-model actors one pose).
            for (const DrawItem& it : g_drawList) {
                double sum = 0.0;
                for (float f : it.bones) sum += f;
                fprintf(stderr, "[SoH3D_GL]   item model=%d boneCount=%d poseSum=%.4f\n", it.modelId,
                        it.boneCount, sum);
            }
        }
    }
    g_drawList.clear();
}

#endif // ENABLE_OPENGL
