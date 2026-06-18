#ifdef ENABLE_VULKAN
#pragma once

#include "gfx_rendering_api.h"
#include "../interpreter.h"

#include <SDL2/SDL.h>
#include <vulkan/vulkan.h>

#include <vector>
#include <map>
#include <cstdint>

namespace Fast {

// Forward declaration of the SDL2 window backend so the Vulkan rendering API can
// pull the SDL_Window out of it (to create the VkSurfaceKHR). gfx_sdl2.cpp is the
// shared window manager for GL / Metal / Vulkan on SDL.
class GfxWindowBackendSDL2;

// Minimal per-combiner shader record. For Milestone 1 (clear only) we don't build
// pipelines yet; we just track the combiner feature flags the interpreter queries
// via ShaderGetInfo so the draw-prep path doesn't dereference null. The real
// pipeline + SPIR-V live here in Milestone 2.
//
// Named *Vulkan (not the interface's opaque Fast::ShaderProgram) because multiple
// backends compile in the same build (GL + Vulkan on Linux); only one header may
// define Fast::ShaderProgram (gfx_opengl.h does). We treat ShaderProgram* opaquely
// and cast, exactly like gfx_metal's ShaderProgramMetal.
struct ShaderProgramVulkan {
    uint8_t numInputs;
    bool usedTextures[2];
};

class GfxRenderingAPIVulkan : public GfxRenderingAPI {
  public:
    explicit GfxRenderingAPIVulkan(GfxWindowBackendSDL2* windowBackend);
    ~GfxRenderingAPIVulkan() override;

    const char* GetName() override;
    int GetMaxTextureSize() override;
    GfxClipParameters GetClipParameters() override;
    void UnloadShader(ShaderProgram* oldPrg) override;
    void LoadShader(ShaderProgram* newPrg) override;
    void ClearShaderCache() override;
    ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) override;
    ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) override;
    void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) override;
    uint32_t NewTexture() override;
    void SelectTexture(int tile, uint32_t textureId) override;
    void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) override;
    void SetSamplerParameters(int sampler, bool linearFilter, uint32_t cms, uint32_t cmt) override;
    void SetDepthTestAndMask(bool depthTest, bool zUpd) override;
    void SetZmodeDecal(bool decal) override;
    void SetViewport(int x, int y, int width, int height) override;
    void SetScissor(int x, int y, int width, int height) override;
    void SetUseAlpha(bool useAlpha) override;
    void DrawTriangles(float bufVbo[], size_t bufVboLen, size_t bufVboNumTris) override;
    void Init() override;
    void OnResize() override;
    void StartFrame() override;
    void EndFrame() override;
    void FinishRender() override;
    int CreateFramebuffer() override;
    void UpdateFramebufferParameters(int fbId, uint32_t width, uint32_t height, uint32_t msaaLevel, bool openglInvertY,
                                     bool renderTarget, bool hasDepthBuffer, bool canExtractDepth) override;
    void StartDrawToFramebuffer(int fbId, float noiseScale) override;
    void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0,
                         int dstX1, int dstY1) override;
    void ClearFramebuffer(bool color, bool depth) override;
    void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) override;
    void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) override;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fbId, const std::set<std::pair<float, float>>& coordinates) override;
    void* GetFramebufferTextureId(int fbId) override;
    void SelectTextureFb(int fbId) override;
    void DeleteTexture(uint32_t texId) override;
    void SetTextureFilter(FilteringMode mode) override;
    FilteringMode GetTextureFilter() override;
    void SetSrgbMode() override;
    ImTextureID GetTextureById(int id) override;
    void SetCurrentPrimDepth(float depth) override;

  private:
    void CreateInstance();
    void CreateSurface();
    void PickPhysicalDevice();
    void CreateLogicalDevice();
    void CreateSwapchain();
    void CreateRenderPass();
    void CreateFramebuffers();
    void CreateCommandResources();
    void CreateSyncObjects();
    void RecreateSwapchain();
    void DestroySwapchain();
    // Milestone-1 verification: copy the just-rendered swapchain image to a host
    // buffer and write a top-down PPM. Honors the same SOH_FRAMEDUMP env + REPL
    // on-demand dump triggers as the GL window-backend path.
    void MaybeDumpFrame();
    void WriteSwapchainPpm(const char* path);

    GfxWindowBackendSDL2* mWindowBackend = nullptr;
    SDL_Window* mWindow = nullptr;

    VkInstance mInstance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR mSurface = VK_NULL_HANDLE;
    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties mPhysicalDeviceProps{};
    VkDevice mDevice = VK_NULL_HANDLE;
    uint32_t mGraphicsQueueFamily = UINT32_MAX;
    uint32_t mPresentQueueFamily = UINT32_MAX;
    VkQueue mGraphicsQueue = VK_NULL_HANDLE;
    VkQueue mPresentQueue = VK_NULL_HANDLE;

    VkSwapchainKHR mSwapchain = VK_NULL_HANDLE;
    VkFormat mSwapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D mSwapchainExtent{};
    std::vector<VkImage> mSwapchainImages;
    std::vector<VkImageView> mSwapchainImageViews;
    std::vector<VkFramebuffer> mSwapchainFramebuffers;
    VkRenderPass mRenderPass = VK_NULL_HANDLE;

    VkCommandPool mCommandPool = VK_NULL_HANDLE;

    static constexpr int kMaxFramesInFlight = 2;
    std::vector<VkCommandBuffer> mCommandBuffers;
    std::vector<VkSemaphore> mImageAvailableSemaphores;
    std::vector<VkSemaphore> mRenderFinishedSemaphores;
    std::vector<VkFence> mInFlightFences;
    uint32_t mCurrentFrame = 0;
    uint32_t mImageIndex = 0;
    bool mFrameAcquired = false;
    bool mEnableValidation = false;

    // Clear color for the current frame's screen framebuffer. A distinctive teal
    // (not black) so M1 verification can confirm Vulkan actually drove the present.
    float mClearColor[4] = { 0.10f, 0.35f, 0.45f, 1.0f };

    // Combiner-shader records keyed by (id0,id1). Populated lazily; no GPU pipeline
    // in Milestone 1.
    std::map<std::pair<uint64_t, uint64_t>, ShaderProgramVulkan> mShaderProgramPool;
    FilteringMode mCurrentFilterMode = FILTER_THREE_POINT;
    uint32_t mNextTextureId = 1;
    int mFramebufferCount = 1; // id 0 = screen
};
} // namespace Fast

#endif
