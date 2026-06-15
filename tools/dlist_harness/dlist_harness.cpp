// SoH3D dlist render harness — the SoH-side counterpart to the Azahar decode
// oracle. It drives libultraship's Fast3D interpreter (the REAL render path)
// over a generated CMB->F3DEX2 display list, WITHOUT booting the game or opening
// a window.
//
// Two modes:
//   (default) RECORDING  — a no-GPU GfxRenderingAPI that just logs every texture
//                          LOAD / UPLOAD / triangle draw. Answers "does LUS upload
//                          this model's texture?" in milliseconds. No GL context.
//   --gl                 — the REAL GfxRenderingAPIOGL rasterising into an offscreen
//                          FBO via an EGL *surfaceless* OpenGL context (llvmpipe,
//                          no X server, no Xvfb, fully deterministic software GL),
//                          then glGetTexImage -> PPM. This is the true both-renderers
//                          pixel A/B counterpart to the Azahar oracle render.
//
// Build:  cmake -S Shipwright -B <build> -DLUS_BUILD_DLIST_HARNESS=ON
//         cmake --build <build> --target soh3d_dlist_harness
// Run:    soh3d_dlist_harness                 (recording mode)
//         soh3d_dlist_harness --gl [--out scratch/render/kibako_lus.ppm] [--size 640x480]
//
// GL mode needs the shader archive (shaders/opengl/default.shader.glsl lives in
// soh.o2r) mounted via the ResourceManager — pass --o2r <path> or set SOH3D_O2R;
// it otherwise probes a few standard build locations.
//
// It links the generated soh3d_kibako_model.c directly (raw Vtx[]/Gfx[]/tex
// arrays, no ResourceManager needed for the geometry itself).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>

#include <ship/Context.h>
#include <fast/interpreter.h>
#include <fast/debug/GfxDebugger.h>
#include <fast/backends/gfx_rendering_api.h>
#include <fast/backends/gfx_window_manager_api.h>
#include <fast/backends/gfx_opengl.h> // GfxRenderingAPIOGL + GL prototypes (SDL_opengl.h on Linux)
#include <libultraship/libultra/gbi.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

// The generated model under test (raw C arrays). Declared extern; linked in.
extern "C" {
extern Gfx soh3d_kibako_model_dl[];
}

namespace Fast {
// Free function in interpreter.cpp (Fast namespace) that caches the instance the
// command handlers reach through mInstance.lock().
void GfxSetInstance(std::shared_ptr<Interpreter> gfx);
} // namespace Fast

using namespace Fast;

// ---------------------------------------------------------------------------
// Recording rendering API: no GPU, just logs the calls we care about.
// ---------------------------------------------------------------------------
class RecordingRenderingAPI : public GfxRenderingAPI {
  public:
    uint32_t mTexCounter = 1;
    int mFbCounter = 1;
    int mCurrentTile = -1;
    uint32_t mUploadCount = 0;
    uint32_t mTriDrawCount = 0;

    const char* GetName() override {
        return "recording";
    }
    int GetMaxTextureSize() override {
        return 16384;
    }
    GfxClipParameters GetClipParameters() override {
        return { false, false };
    }
    void UnloadShader(ShaderProgram*) override {
    }
    void LoadShader(ShaderProgram*) override {
    }
    void ClearShaderCache() override {
    }
    // Return a non-null dummy; the interpreter only stores it and asks us about
    // it via ShaderGetInfo (which we answer), it never dereferences it.
    ShaderProgram* CreateAndLoadNewShader(uint64_t, uint64_t) override {
        return reinterpret_cast<ShaderProgram*>(&mShaderDummy);
    }
    ShaderProgram* LookupShader(uint64_t, uint64_t) override {
        return nullptr;
    }
    void ShaderGetInfo(ShaderProgram*, uint8_t* numInputs, bool usedTextures[2]) override {
        *numInputs = 1;
        usedTextures[0] = true;
        usedTextures[1] = false;
    }
    uint32_t NewTexture() override {
        return mTexCounter++;
    }
    void SelectTexture(int tile, uint32_t textureId) override {
        mCurrentTile = tile;
    }
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override {
        mUploadCount++;
        printf("[HARNESS upload] #%u tile=%d %ux%u (%u px) first=%d,%d,%d,%d\n", mUploadCount, mCurrentTile, width,
               height, width * height, rgba32Buf[0], rgba32Buf[1], rgba32Buf[2], rgba32Buf[3]);
        fflush(stdout);
    }
    void SetSamplerParameters(int, bool, uint32_t, uint32_t) override {
    }
    void SetDepthTestAndMask(bool, bool) override {
    }
    void SetZmodeDecal(bool) override {
    }
    void SetViewport(int, int, int, int) override {
    }
    void SetScissor(int, int, int, int) override {
    }
    void SetUseAlpha(bool) override {
    }
    void DrawTriangles(float[], size_t, size_t buf_vbo_num_tris) override {
        mTriDrawCount += (uint32_t)buf_vbo_num_tris;
    }
    void Init() override {
    }
    void OnResize() override {
    }
    void StartFrame() override {
    }
    void EndFrame() override {
    }
    void FinishRender() override {
    }
    int CreateFramebuffer() override {
        return mFbCounter++;
    }
    void UpdateFramebufferParameters(int, uint32_t, uint32_t, uint32_t, bool, bool, bool, bool) override {
    }
    void StartDrawToFramebuffer(int, float) override {
    }
    void CopyFramebuffer(int, int, int, int, int, int, int, int, int, int) override {
    }
    void ClearFramebuffer(bool, bool) override {
    }
    void ReadFramebufferToCPU(int, uint32_t, uint32_t, uint16_t*) override {
    }
    void ResolveMSAAColorBuffer(int, int) override {
    }
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int, const std::set<std::pair<float, float>>&) override {
        return {};
    }
    void* GetFramebufferTextureId(int) override {
        return nullptr;
    }
    void SelectTextureFb(int) override {
    }
    void DeleteTexture(uint32_t) override {
    }
    void SetTextureFilter(FilteringMode) override {
    }
    FilteringMode GetTextureFilter() override {
        return FILTER_NONE;
    }
    void SetSrgbMode() override {
    }
    ImTextureID GetTextureById(int) override {
        return (ImTextureID)0;
    }
    void SetCurrentPrimDepth(float) override {
    }

  private:
    int mShaderDummy = 0;
};

// ---------------------------------------------------------------------------
// No-op window backend: the interpreter only needs Init + GetDimensions for the
// headless render path. (The EGL context is created by us, not by this backend.)
// ---------------------------------------------------------------------------
class StubWindowBackend : public GfxWindowBackend {
  public:
    uint32_t mW = 640, mH = 480;
    void Init(const char*, const char*, bool, uint32_t, uint32_t, int32_t, int32_t) override {
    }
    void Close() override {
    }
    void SetKeyboardCallbacks(bool (*)(int), bool (*)(int), void (*)()) override {
    }
    void SetMouseCallbacks(bool (*)(int), bool (*)(int)) override {
    }
    void SetFullscreenChangedCallback(void (*)(bool)) override {
    }
    void SetFullscreen(bool) override {
    }
    void GetActiveWindowRefreshRate(uint32_t* r) override {
        *r = 60;
    }
    void SetCursorVisibility(bool) override {
    }
    void SetMousePos(int32_t, int32_t) override {
    }
    void GetMousePos(int32_t* x, int32_t* y) override {
        *x = 0;
        *y = 0;
    }
    void GetMouseDelta(int32_t* x, int32_t* y) override {
        *x = 0;
        *y = 0;
    }
    void GetMouseWheel(float* x, float* y) override {
        *x = 0;
        *y = 0;
    }
    bool GetMouseState(uint32_t) override {
        return false;
    }
    void SetMouseCapture(bool) override {
    }
    bool IsMouseCaptured() override {
        return false;
    }
    void GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY) override {
        *width = mW;
        *height = mH;
        *posX = 0;
        *posY = 0;
    }
    void SetDimensions(uint32_t, uint32_t, int32_t, int32_t) override {
    }
    Ship::WindowRect GetPrimaryMonitorRect() override {
        return { 0, 0, (int32_t)mW, (int32_t)mH };
    }
    void HandleEvents() override {
    }
    bool IsFrameReady() override {
        return true;
    }
    void SwapBuffersBegin() override {
    }
    void SwapBuffersEnd() override {
    }
    double GetTime() override {
        return 0.0;
    }
    int GetTargetFps() override {
        return 60;
    }
    void SetTargetFps(int) override {
    }
    void SetMaxFrameLatency(int) override {
    }
    const char* GetKeyName(int) override {
        return "";
    }
    bool CanDisableVsync() override {
        return true;
    }
    bool IsRunning() override {
        return true;
    }
    void Destroy() override {
    }
    bool IsFullscreen() override {
        return false;
    }
};

// ---------------------------------------------------------------------------
// EGL surfaceless OpenGL context (no window, no X server). Renders into FBOs only.
// ---------------------------------------------------------------------------
static EGLDisplay g_eglDpy = EGL_NO_DISPLAY;
static EGLContext g_eglCtx = EGL_NO_CONTEXT;

static bool EglInitSurfaceless() {
    g_eglDpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    if (g_eglDpy == EGL_NO_DISPLAY) {
        fprintf(stderr, "[HARNESS] eglGetPlatformDisplay(SURFACELESS_MESA) failed\n");
        return false;
    }
    EGLint major = 0, minor = 0;
    if (!eglInitialize(g_eglDpy, &major, &minor)) {
        fprintf(stderr, "[HARNESS] eglInitialize failed (0x%x)\n", eglGetError());
        return false;
    }
    printf("[HARNESS] EGL %d.%d surfaceless; vendor=%s\n", major, minor, eglQueryString(g_eglDpy, EGL_VENDOR));

    if (!eglBindAPI(EGL_OPENGL_API)) {
        fprintf(stderr, "[HARNESS] eglBindAPI(OPENGL) failed\n");
        return false;
    }

    // The Mesa surfaceless platform advertises ZERO EGLConfigs (there is no native
    // window system to describe). Create a config-less context instead, via
    // EGL_KHR_no_config_context — we only ever render into FBOs, where the config's
    // colour/depth format is irrelevant.
    const char* ext = eglQueryString(g_eglDpy, EGL_EXTENSIONS);
    if (!ext || !strstr(ext, "EGL_KHR_no_config_context") || !strstr(ext, "EGL_KHR_surfaceless_context")) {
        fprintf(stderr, "[HARNESS] EGL lacks no_config_context / surfaceless_context\n");
        return false;
    }
    EGLConfig cfg = EGL_NO_CONFIG_KHR;

    // Compatibility profile: the GLSL the OGL backend emits on desktop Linux is
    // #version 130 (varying / gl_FragColor / texture2D) and it draws without a VAO
    // — both require a compatibility (non-core) context. This mirrors the de-facto
    // context SoH gets on Linux.
    const EGLint ctxAttribs[] = { EGL_CONTEXT_MAJOR_VERSION,
                                  3,
                                  EGL_CONTEXT_MINOR_VERSION,
                                  3,
                                  EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                  EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
                                  EGL_NONE };
    g_eglCtx = eglCreateContext(g_eglDpy, cfg, EGL_NO_CONTEXT, ctxAttribs);
    if (g_eglCtx == EGL_NO_CONTEXT) {
        fprintf(stderr, "[HARNESS] eglCreateContext failed (0x%x)\n", eglGetError());
        return false;
    }
    // Surfaceless: no draw/read surface; the backend only ever binds FBOs.
    if (!eglMakeCurrent(g_eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, g_eglCtx)) {
        fprintf(stderr, "[HARNESS] eglMakeCurrent(surfaceless) failed (0x%x)\n", eglGetError());
        return false;
    }
    printf("[HARNESS] GL_RENDERER=%s  GL_VERSION=%s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
    return true;
}

static void EglShutdown() {
    if (g_eglDpy != EGL_NO_DISPLAY) {
        eglMakeCurrent(g_eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (g_eglCtx != EGL_NO_CONTEXT)
            eglDestroyContext(g_eglDpy, g_eglCtx);
        eglTerminate(g_eglDpy);
    }
}

// Write an RGB8 buffer (GL bottom-left origin) to a top-to-bottom PPM (P6).
static bool WritePpmFlipped(const std::string& path, const uint8_t* rgb, uint32_t w, uint32_t h) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[HARNESS] cannot open %s for write\n", path.c_str());
        return false;
    }
    fprintf(f, "P6\n%u %u\n255\n", w, h);
    for (int y = (int)h - 1; y >= 0; y--) // flip: GL row 0 is the bottom
        fwrite(rgb + (size_t)y * w * 3, 1, (size_t)w * 3, f);
    fclose(f);
    return true;
}

// Locate the shader archive (soh.o2r) for GL mode.
static std::string FindO2r(const char* explicitPath) {
    if (explicitPath && *explicitPath && std::filesystem::exists(explicitPath))
        return explicitPath;
    if (const char* env = getenv("SOH3D_O2R"); env && *env && std::filesystem::exists(env))
        return env;
    const char* candidates[] = {
        "Shipwright/build-cmake/soh/soh.o2r", "build-cmake/soh/soh.o2r", "../../soh/soh.o2r", "soh.o2r",
    };
    for (const char* c : candidates)
        if (std::filesystem::exists(c))
            return c;
    return {};
}

// ---------------------------------------------------------------------------
// Build the command stream shared by both modes: relocate the model's texture
// pointers high, then a prologue that establishes a viewport / scissor / PRIM
// colour / projection+modelview so the crate actually rasterises on-screen.
// ---------------------------------------------------------------------------
struct BuiltDlist {
    std::vector<Gfx> model;   // relocated copy of the model dlist
    std::vector<Gfx> dl;      // prologue + call(model) + end
    Vp vp{};                  // referenced by gsSPViewport (must outlive Run)
    Mtx projMtx{}, mvMtx{};   // mtx_replacement keys (addresses matter, contents unused)
    std::unordered_map<Mtx*, MtxF> mtxReplacements;
};

static void BuildDlist(BuiltDlist& b) {
    // Copy the model dlist into a mutable buffer (up to and including G_ENDDL).
    for (int i = 0;; i++) {
        b.model.push_back(soh3d_kibako_model_dl[i]);
        if ((uint8_t)(soh3d_kibako_model_dl[i].words.w0 >> 24) == 0xDF) // G_ENDDL
            break;
    }

    // Relocate every G_SETTIMG (0xFD) texture pointer into a high mmap'd buffer.
    // HARNESS FIXUP, not a workaround for any bug under test: in the real soh.elf
    // (a large PIE) these static textures live at high addresses, so the
    // interpreter's "addr <= 0x0FFFFFFF => unresolved N64 segment, skip" guard in
    // gfx_set_timg_handler_rdp passes. In this small standalone binary the same
    // static data sits at ~6 MB and is falsely rejected, so we copy it high to
    // faithfully exercise the in-game path.
    for (size_t i = 0; i + 1 < b.model.size(); i++) {
        if ((uint8_t)(b.model[i].words.w0 >> 24) != 0xFD) // G_SETTIMG
            continue;
        uint32_t siz = (b.model[i].words.w0 >> 19) & 0x3; // 0:4b 1:8b 2:16b 3:32b
        uint32_t texels = 0;
        for (size_t j = i + 1; j < b.model.size(); j++) {
            if ((uint8_t)(b.model[j].words.w0 >> 24) == 0x47) { // LoadBlockWide
                texels = (uint32_t)(b.model[j].words.w1 & 0xFFFFFFFF) + 1;
                break;
            }
        }
        size_t bytes = (siz == 0) ? (texels + 1) / 2 : (siz == 1) ? texels : (siz == 2) ? texels * 2 : texels * 4;
        const uint8_t* orig = (const uint8_t*)(uintptr_t)b.model[i].words.w1;
        void* hi = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        memcpy(hi, orig, bytes);
        printf("[HARNESS] relocated G_SETTIMG[%zu] tex %p -> %p (%zu bytes, siz=%u)\n", i, (void*)orig, hi, bytes, siz);
        b.model[i].words.w1 = (uintptr_t)hi;
    }

    // Identity projection; modelview maps the model (x:[-300,300] y:[0,480] z~240)
    // into NDC [-1,1]. Convention (GfxSpVertex): clip_j = sum_k ob[k]*MP[k][j] +
    // MP[3][j], so translation lives in row [3][*].
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            b.mtxReplacements[&b.projMtx].mf[i][j] = (i == j) ? 1.0f : 0.0f;
    MtxF& mv = b.mtxReplacements[&b.mvMtx];
    memset(mv.mf, 0, sizeof(mv.mf));
    mv.mf[0][0] = 1.0f / 800.0f;
    mv.mf[1][1] = 1.0f / 400.0f;
    mv.mf[2][2] = 1.0f / 2000.0f;
    mv.mf[3][1] = -240.0f / 400.0f;
    mv.mf[3][3] = 1.0f;

    // Standard full-screen 320x240 native viewport (scale = half-dim*4).
    b.vp.vp.vscale[0] = (SCREEN_WIDTH / 2) * 4;
    b.vp.vp.vscale[1] = (SCREEN_HEIGHT / 2) * 4;
    b.vp.vp.vscale[2] = G_MAXZ;
    b.vp.vp.vscale[3] = 0;
    b.vp.vp.vtrans[0] = (SCREEN_WIDTH / 2) * 4;
    b.vp.vp.vtrans[1] = (SCREEN_HEIGHT / 2) * 4;
    b.vp.vp.vtrans[2] = 0;
    b.vp.vp.vtrans[3] = 0;

    // Prologue: viewport + scissor + PRIM(white) + matrices, then call the model.
    // PRIM matters: the crate combiner is MODULATE x PRIM, so PRIM=0 => black.
    // (The gs* macros expand to brace-aggregate initializers, so assign to named
    // locals — a C-style (Gfx){...} compound-literal cast is not valid C++.)
    Gfx cViewport = gsSPViewport(&b.vp);
    Gfx cScissor = gsDPSetScissor(G_SC_NON_INTERLACE, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
    Gfx cProj = gsSPMatrix(&b.projMtx, G_MTX_NOPUSH | G_MTX_LOAD | G_MTX_PROJECTION);
    Gfx cMv = gsSPMatrix(&b.mvMtx, G_MTX_NOPUSH | G_MTX_LOAD | G_MTX_MODELVIEW);
    Gfx cPrim = gsDPSetPrimColor(0, 0, 255, 255, 255, 255);
    Gfx cCall = gsSPDisplayList(b.model.data());
    Gfx cEnd = gsSPEndDisplayList();
    b.dl.push_back(cViewport);
    b.dl.push_back(cScissor);
    b.dl.push_back(cProj);
    b.dl.push_back(cMv);
    b.dl.push_back(cPrim);
    b.dl.push_back(cCall);
    b.dl.push_back(cEnd);
}

int main(int argc, char** argv) {
    bool glMode = false;
    std::string outPath = "scratch/render/kibako_lus.ppm";
    std::string o2rArg;
    uint32_t W = 640, H = 480;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gl")
            glMode = true;
        else if (a == "--out" && i + 1 < argc)
            outPath = argv[++i];
        else if (a == "--o2r" && i + 1 < argc)
            o2rArg = argv[++i];
        else if (a == "--size" && i + 1 < argc) {
            unsigned w, h;
            if (sscanf(argv[++i], "%ux%u", &w, &h) == 2) {
                W = w;
                H = h;
            }
        }
    }

    auto* ctx = Ship::Context::CreateUninitializedInstance("soh3d_harness", "soh3d_harness", "");
    ctx->InitLogging();
    ctx->InitConfiguration();
    ctx->InitConsoleVariables();

    std::string o2r;
    if (glMode) {
        o2r = FindO2r(o2rArg.c_str());
        if (o2r.empty()) {
            fprintf(stderr, "[HARNESS] GL mode needs the shader archive (soh.o2r). "
                            "Pass --o2r <path> or set SOH3D_O2R.\n");
            return 2;
        }
        // ResourceManager mounts soh.o2r so the OGL backend can load
        // shaders/opengl/default.shader.glsl (compiled per-combiner at draw time).
        ctx->InitResourceManager({ o2r });
        printf("[HARNESS] mounted shader archive: %s\n", o2r.c_str());
        if (!EglInitSurfaceless())
            return 3;
    }

    auto rec = std::make_unique<RecordingRenderingAPI>();
    auto ogl = glMode ? std::make_unique<GfxRenderingAPIOGL>() : nullptr;
    GfxRenderingAPI* rapi = glMode ? (GfxRenderingAPI*)ogl.get() : (GfxRenderingAPI*)rec.get();

    auto wapi = std::make_unique<StubWindowBackend>();
    wapi->mW = W;
    wapi->mH = H;

    auto gfx = std::make_shared<Interpreter>();
    GfxSetInstance(gfx);
    gfx->SetGfxDebugger(std::make_shared<GfxDebugger>());
    gfx->Init(wapi.get(), rapi, "soh3d_harness", false, W, H, 0, 0);

    BuiltDlist b;
    BuildDlist(b);

    printf("[HARNESS] running crate dlist through LUS interpreter (%s, %ux%u)...\n", glMode ? "GL" : "recording", W, H);
    fflush(stdout);
    gfx->StartFrame();
    gfx->Run(b.dl.data(), b.mtxReplacements);

    if (glMode) {
        glFinish();
        // mGameFb is deterministically the FIRST framebuffer the interpreter creates
        // in Init() (rapi->Init() reserves index 0 for the screen; Init() then calls
        // CreateFramebuffer() twice -> mGameFb=1, mGameFbMsaaResolved=2). With the
        // default MSAA=1, fb1's colour attachment is a plain RGB8 texture, which is
        // what Run() leaves the rendered crate in (fb 0 is re-cleared at frame end).
        const int kGameFb = 1;
        GLuint tex = (GLuint)(uintptr_t)rapi->GetFramebufferTextureId(kGameFb);
        std::vector<uint8_t> rgb((size_t)W * H * 3);
        glBindTexture(GL_TEXTURE_2D, tex);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        // Quick non-black pixel count so the log carries a quantitative signal.
        size_t nonBlack = 0;
        for (size_t i = 0; i < rgb.size(); i += 3)
            if (rgb[i] | rgb[i + 1] | rgb[i + 2])
                nonBlack++;
        if (WritePpmFlipped(outPath, rgb.data(), W, H))
            printf("[HARNESS] wrote %s (%ux%u, %zu/%u non-black px)\n", outPath.c_str(), W, H, nonBlack, W * H);
        EglShutdown();
    } else {
        printf("[HARNESS] done: %u UploadTexture call(s), %u triangle(s) drawn.\n", rec->mUploadCount,
               rec->mTriDrawCount);
    }
    fflush(stdout);
    return 0;
}
