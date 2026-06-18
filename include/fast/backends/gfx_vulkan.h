#ifdef ENABLE_VULKAN
#pragma once

#include "gfx_rendering_api.h"
#include "../interpreter.h"

#include <SDL2/SDL.h>
#include <vulkan/vulkan.h>

#include <vector>
#include <map>
#include <array>
#include <unordered_map>
#include <string>
#include <cstdint>

namespace Fast {

// Forward declaration of the SDL2 window backend so the Vulkan rendering API can
// pull the SDL_Window out of it (to create the VkSurfaceKHR). gfx_sdl2.cpp is the
// shared window manager for GL / Metal / Vulkan on SDL.
class GfxWindowBackendSDL2;

// Per-combiner shader record. Holds the compiled SPIR-V shader modules and the
// vertex-input layout derived from the color-combiner features, mirroring the GL
// backend's ShaderProgram. VkPipelines are built lazily (per render-state combo)
// and cached separately in mPipelineCache, keyed on (id0,id1,stateBits).
//
// Named *Vulkan (not the interface's opaque Fast::ShaderProgram) because multiple
// backends compile in the same build (GL + Vulkan on Linux); only one header may
// define Fast::ShaderProgram (gfx_opengl.h does). We treat ShaderProgram* opaquely
// and cast, exactly like gfx_metal's ShaderProgramMetal.
struct ShaderProgramVulkan {
    uint64_t id0 = 0, id1 = 0;
    uint8_t numInputs = 0;
    bool usedTextures[2] = { false, false }; // tex0, tex1
    bool usedMasks[2] = { false, false };
    bool usedBlend[2] = { false, false };
    uint8_t numFloats = 0; // vertex stride in floats
    VkShaderModule vert = VK_NULL_HANDLE;
    VkShaderModule frag = VK_NULL_HANDLE;
    // Which of the 6 sampler slots (tex0,tex1,mask0,mask1,blend0,blend1) are used.
    bool usedSlot[6] = { false, false, false, false, false, false };
    // Vertex input attributes in declaration order (location == index).
    struct Attr {
        uint32_t size;   // component count (1..4 floats)
        uint32_t offset; // byte offset within the vertex
    };
    std::vector<Attr> attribs;
};

// Pipeline cache key: a shader (combiner) plus the render-state knobs that affect
// pipeline creation (depth, blend, polygon offset). Viewport/scissor/depth-bias
// magnitude are dynamic state, so they are NOT part of the key.
struct VulkanPipelineKey {
    uint64_t id0, id1;
    uint32_t stateBits; // bit0 depthTest, bit1 depthMask, bit2 zmodeDecal, bit3 useAlpha
    bool operator==(const VulkanPipelineKey& o) const {
        return id0 == o.id0 && id1 == o.id1 && stateBits == o.stateBits;
    }
};
struct VulkanPipelineKeyHash {
    size_t operator()(const VulkanPipelineKey& k) const {
        return std::hash<uint64_t>()(k.id0) ^ (std::hash<uint64_t>()(k.id1) << 1) ^
               (std::hash<uint32_t>()(k.stateBits) << 2);
    }
};

// A GPU texture (VkImage) plus its sampling metadata.
struct TextureVulkan {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    uint16_t filtering = 0; // FILTER_* (for the in-shader three-point path)
    bool linearFilter = false;
    uint32_t cms = 0, cmt = 0;
    bool uploaded = false;
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
    void CreatePerImageSync();
    void DestroyPerImageSync();
    void RecreateSwapchain();
    void DestroySwapchain();
    // Milestone-1 verification: copy the just-rendered swapchain image to a host
    // buffer and write a top-down PPM. Honors the same SOH_FRAMEDUMP env + REPL
    // on-demand dump triggers as the GL window-backend path.
    void MaybeDumpFrame();
    void WriteSwapchainPpm(const char* path);

    // ---- M2: real rendering (pipelines / vertex streaming / textures) ----
    void CreateDepthResources();
    void DestroyDepthResources();
    void CreateRenderingResources();   // descriptor layout, pipeline layout, per-frame rings, dummy tex
    void DestroyRenderingResources();
    uint32_t FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf,
                      VkDeviceMemory& mem, void** mappedOut);
    void CreateImageRGBA(uint32_t width, uint32_t height, VkImage& image, VkDeviceMemory& mem, VkImageView& view);
    void UploadImageRGBA(VkImage image, uint32_t width, uint32_t height, const uint8_t* rgba);
    std::string BuildVkShaderSource(const struct CCFeatures& cc, bool vertex, ShaderProgramVulkan* prg);
    VkShaderModule CreateShaderModule(const std::vector<uint32_t>& spirv);
    VkPipeline GetOrCreatePipeline(ShaderProgramVulkan* prg, uint32_t stateBits);
    VkSampler GetOrCreateSampler(bool linear, uint32_t cms, uint32_t cmt);
    void BeginFrameRings(); // reset per-frame vbo/ubo/descriptor pool at acquire

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
    std::vector<VkSemaphore> mImageAvailableSemaphores; // per frame-in-flight
    std::vector<VkSemaphore> mRenderFinishedSemaphores; // per SWAPCHAIN IMAGE (present sync)
    std::vector<VkFence> mInFlightFences;               // per frame-in-flight
    std::vector<VkFence> mImagesInFlight;               // per image: fence last submitted for it
    uint32_t mCurrentFrame = 0;
    uint32_t mImageIndex = 0;
    bool mFrameAcquired = false;
    bool mEnableValidation = false;

    // Clear color for the current frame's screen framebuffer. A distinctive teal
    // (not black) so M1 verification can confirm Vulkan actually drove the present.
    float mClearColor[4] = { 0.10f, 0.35f, 0.45f, 1.0f };

    // Depth attachment shared by the swapchain render pass. In M2 every Fast3D draw
    // renders into the swapchain pass (offscreen framebuffers arrive in M3), so a
    // single window-sized depth buffer suffices.
    VkImage mDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory mDepthMemory = VK_NULL_HANDLE;
    VkImageView mDepthView = VK_NULL_HANDLE;
    VkFormat mDepthFormat = VK_FORMAT_D32_SFLOAT;

    // Combiner-shader records keyed by (id0,id1).
    std::map<std::pair<uint64_t, uint64_t>, ShaderProgramVulkan> mShaderProgramPool;
    std::unordered_map<VulkanPipelineKey, VkPipeline, VulkanPipelineKeyHash> mPipelineCache;
    VkDescriptorSetLayout mDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout mPipelineLayout = VK_NULL_HANDLE;

    // Per-frame-in-flight transient resources: a vertex ring, a uniform ring, and a
    // descriptor pool, all reset at the start of each acquired frame.
    struct FrameRing {
        VkBuffer vbo = VK_NULL_HANDLE;
        VkDeviceMemory vboMem = VK_NULL_HANDLE;
        void* vboMapped = nullptr;
        VkDeviceSize vboCapacity = 0;
        VkDeviceSize vboOffset = 0;
        VkBuffer ubo = VK_NULL_HANDLE;
        VkDeviceMemory uboMem = VK_NULL_HANDLE;
        void* uboMapped = nullptr;
        VkDeviceSize uboCapacity = 0;
        VkDeviceSize uboOffset = 0;
        VkDescriptorPool descPool = VK_NULL_HANDLE;
    };
    std::array<FrameRing, kMaxFramesInFlight> mFrameRings;
    VkDeviceSize mUboAlignedSize = 0;

    // Samplers cached by (linear, cms, cmt).
    std::map<uint32_t, VkSampler> mSamplerCache;

    // A 1x1 white texture bound to unused sampler slots so descriptor sets are valid.
    VkImage mDummyImage = VK_NULL_HANDLE;
    VkDeviceMemory mDummyMemory = VK_NULL_HANDLE;
    VkImageView mDummyView = VK_NULL_HANDLE;
    VkSampler mDummySampler = VK_NULL_HANDLE;

    std::vector<TextureVulkan> mTextures; // indexed by texture id

    // Current draw state. (mCurrentDepthTest/Mask/ZmodeDecal, mSrgbMode and
    // mCurrentPrimDepth live in the GfxRenderingAPI base class — do not shadow them.)
    ShaderProgramVulkan* mCurrentShaderProgram = nullptr;
    uint32_t mCurrentTextureIds[6] = { 0, 0, 0, 0, 0, 0 };
    uint8_t mCurrentTile = 0;
    bool mCurrentUseAlpha = false;
    float mCurrentNoiseScale = 0.0f;
    uint32_t mFrameCount = 0;
    VkViewport mCurrentViewport{};
    VkRect2D mCurrentScissor{};

    FilteringMode mCurrentFilterMode = FILTER_THREE_POINT;
    uint32_t mNextTextureId = 1;
    int mFramebufferCount = 1; // id 0 = screen
};
} // namespace Fast

#endif
