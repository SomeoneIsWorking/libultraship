// SoH3D dlist render harness — the SoH-side counterpart to the Azahar decode
// oracle. It drives libultraship's Fast3D interpreter (the REAL render path)
// over a generated CMB->F3DEX2 display list, WITHOUT booting the game, opening a
// window, or touching a GPU. A recording GfxRenderingAPI captures every texture
// LOAD / UPLOAD / triangle draw, so we can answer questions like "does LUS
// actually upload this model's 128x128 RGBA32 texture?" deterministically and in
// milliseconds, instead of navigating a live scene where the actor may be culled.
//
// Build:  cmake --build <build> --target soh3d_dlist_harness
//         (configure libultraship with -DLUS_BUILD_DLIST_HARNESS=ON)
// Run:    soh3d_dlist_harness            (runs the built-in crate model)
//
// It links the generated soh3d_kibako_model.c directly (raw Vtx[]/Gfx[]/tex
// arrays, no ResourceManager needed).

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>

#include <ship/Context.h>
#include <fast/interpreter.h>
#include <fast/debug/GfxDebugger.h>
#include <fast/backends/gfx_rendering_api.h>
#include <fast/backends/gfx_window_manager_api.h>
#include <libultraship/libultra/gbi.h>

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
// headless render path.
// ---------------------------------------------------------------------------
class StubWindowBackend : public GfxWindowBackend {
  public:
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
        *width = 1280;
        *height = 720;
        *posX = 0;
        *posY = 0;
    }
    void SetDimensions(uint32_t, uint32_t, int32_t, int32_t) override {
    }
    Ship::WindowRect GetPrimaryMonitorRect() override {
        return { 0, 0, 1280, 720 };
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

int main() {
    // Minimal Context: the interpreter only needs ConsoleVariables non-null.
    auto* ctx = Ship::Context::CreateUninitializedInstance("soh3d_harness", "soh3d_harness", "");
    ctx->InitLogging();
    ctx->InitConfiguration();
    ctx->InitConsoleVariables();

    auto rapi = std::make_unique<RecordingRenderingAPI>();
    auto wapi = std::make_unique<StubWindowBackend>();

    auto gfx = std::make_shared<Interpreter>();
    GfxSetInstance(gfx);
    gfx->SetGfxDebugger(std::make_shared<GfxDebugger>());
    gfx->Init(wapi.get(), rapi.get(), "soh3d_harness", false, 1280, 720, 0, 0);

    // Project the model (x:[-300,300] y:[0,480] z~240) into NDC [-1,1] with w=1.
    // Convention (GfxSpVertex): clip_j = sum_k ob[k]*MP[k][j] + MP[3][j], so the
    // translation lives in row [3][*]. Diagonal scale + a y-centering translate.
    Mtx projMtx{};
    Mtx mvMtx{};
    MtxF projF{};
    MtxF mvF{};
    // identity projection
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            projF.mf[i][j] = (i == j) ? 1.0f : 0.0f;
    // modelview: x/800, (y-240)/400, z/2000, w=1
    memset(mvF.mf, 0, sizeof(mvF.mf));
    mvF.mf[0][0] = 1.0f / 800.0f;
    mvF.mf[1][1] = 1.0f / 400.0f;
    mvF.mf[2][2] = 1.0f / 2000.0f;
    mvF.mf[3][1] = -240.0f / 400.0f;
    mvF.mf[3][3] = 1.0f;

    std::unordered_map<Mtx*, MtxF> mtxReplacements;
    mtxReplacements[&projMtx] = projF;
    mtxReplacements[&mvMtx] = mvF;

    // Copy the model dlist into a mutable buffer (up to and including G_ENDDL).
    std::vector<Gfx> model;
    for (int i = 0;; i++) {
        model.push_back(soh3d_kibako_model_dl[i]);
        if ((uint8_t)(soh3d_kibako_model_dl[i].words.w0 >> 24) == 0xDF) // G_ENDDL
            break;
    }

    // Relocate every G_SETTIMG (0xFD) texture pointer into a high mmap'd buffer.
    // This is a HARNESS FIXUP, not a workaround for the bug under test: in the real
    // soh.elf (a large PIE binary) these static textures live at high addresses, so
    // the interpreter's "i <= 0x0FFFFFFF => unresolved N64 segment, skip" guard in
    // gfx_set_timg_handler_rdp passes. In this small standalone binary the same
    // static data sits at ~6 MB and is falsely rejected, so we copy it high to
    // faithfully exercise the in-game path.
    for (size_t i = 0; i + 1 < model.size(); i++) {
        if ((uint8_t)(model[i].words.w0 >> 24) != 0xFD) // G_SETTIMG
            continue;
        // siz = bits[19:20]; bytes/texel: 4b->.5, 8b->1, 16b->2, 32b->4
        uint32_t siz = (model[i].words.w0 >> 19) & 0x3;
        // Find the following LoadBlockWide (0x47) for the texel count.
        uint32_t texels = 0;
        for (size_t j = i + 1; j < model.size(); j++) {
            if ((uint8_t)(model[j].words.w0 >> 24) == 0x47) {
                texels = (uint32_t)(model[j].words.w1 & 0xFFFFFFFF) + 1;
                break;
            }
        }
        size_t bytes = (siz == 0) ? (texels + 1) / 2 : texels * (1u << (siz - 1)) * (siz == 3 ? 1 : 1);
        if (siz == 3)
            bytes = (size_t)texels * 4;
        else if (siz == 2)
            bytes = (size_t)texels * 2;
        else if (siz == 1)
            bytes = texels;
        const uint8_t* orig = (const uint8_t*)(uintptr_t)model[i].words.w1;
        void* hi = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        memcpy(hi, orig, bytes);
        printf("[HARNESS] relocated G_SETTIMG[%zu] tex %p -> %p (%zu bytes, siz=%u)\n", i, (void*)orig, hi, bytes, siz);
        model[i].words.w1 = (uintptr_t)hi;
    }

    // Dump the raw opcode stream (post-relocation) so we can see each command.
    printf("[HARNESS] model dlist opcode stream:\n");
    for (size_t i = 0; i < model.size() && i < 24; i++) {
        uint8_t op = (uint8_t)(model[i].words.w0 >> 24);
        printf("  [%2zu] op=0x%02X w0=0x%016lx w1=0x%016lx\n", i, op, (unsigned long)model[i].words.w0,
               (unsigned long)model[i].words.w1);
        if (op == 0xDF)
            break;
    }
    fflush(stdout);

    // Prologue: load projection + modelview, then call the (relocated) model dlist.
    std::vector<Gfx> dl;
    Gfx setProj = gsSPMatrix(&projMtx, G_MTX_NOPUSH | G_MTX_LOAD | G_MTX_PROJECTION);
    Gfx setMv = gsSPMatrix(&mvMtx, G_MTX_NOPUSH | G_MTX_LOAD | G_MTX_MODELVIEW);
    Gfx callModel = gsSPDisplayList(model.data());
    Gfx endDl = gsSPEndDisplayList();
    dl.push_back(setProj);
    dl.push_back(setMv);
    dl.push_back(callModel);
    dl.push_back(endDl);

    printf("[HARNESS] running crate dlist through LUS interpreter...\n");
    fflush(stdout);
    gfx->StartFrame();
    gfx->Run(dl.data(), mtxReplacements);

    printf("[HARNESS] done: %u UploadTexture call(s), %u triangle(s) drawn.\n", rapi->mUploadCount,
           rapi->mTriDrawCount);
    fflush(stdout);
    return 0;
}
