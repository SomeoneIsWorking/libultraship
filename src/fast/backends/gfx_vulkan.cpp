#include "ship/window/Window.h"
#ifdef ENABLE_VULKAN

// ============================================================================
// Vulkan rendering backend for Fast3D (soh3d).
//
// Milestone 1 scope: stand up the full Vulkan context (instance / device /
// queues / swapchain / render pass / command + sync objects), integrate it into
// the engine frame loop, and CLEAR the screen each frame. All draw/texture/
// framebuffer entry points are intentionally inert here — they arrive in later
// milestones (M2 pipelines+textures, M3 framebuffers, M5 soh3d pass). The point
// of M1 is to prove the window/surface/present plumbing works on Linux (RADV)
// before any rendering logic is ported.
//
// Frame-loop mapping (see Interpreter::EndFrame):
//   mRapi->StartFrame()    -> acquire swapchain image, begin cmd buffer + render
//                             pass (idempotent: the interpreter calls StartFrame
//                             twice per displayed frame).
//   mRapi->EndFrame()      -> end render pass + command buffer.
//   mWapi->SwapBuffersBegin-> (window) framerate sync only for Vulkan.
//   mRapi->FinishRender()  -> submit + present, then reset frame state.
// ============================================================================

#include "fast/backends/gfx_vulkan.h"
#include "fast/backends/gfx_sdl.h"
#include "fast/interpreter.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"

#include <SDL2/SDL_vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <spdlog/spdlog.h>

// SoH3D frame-dump globals (defined in gfx_sdl2.cpp). The REPL sets these to
// capture the current frame on demand; we honor them from the Vulkan present path
// since glReadPixels is unavailable here.
extern "C" {
extern char gSoh3dDumpPath[1024];
extern volatile int gSoh3dDumpPending;
}

namespace Fast {

#define VK_CHECK(expr)                                                                                                 \
    do {                                                                                                               \
        VkResult vk_check_res_ = (expr);                                                                               \
        if (vk_check_res_ != VK_SUCCESS) {                                                                             \
            SPDLOG_ERROR("Vulkan call failed ({}): {} = {}", __LINE__, #expr, (int)vk_check_res_);                     \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

GfxRenderingAPIVulkan::GfxRenderingAPIVulkan(GfxWindowBackendSDL2* windowBackend) : mWindowBackend(windowBackend) {
    const char* val = getenv("SOH3D_VK_VALIDATION");
    mEnableValidation = val != nullptr && val[0] == '1';
}

GfxRenderingAPIVulkan::~GfxRenderingAPIVulkan() {
    if (mDevice != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(mDevice);
        DestroySwapchain();
        for (int i = 0; i < kMaxFramesInFlight; i++) {
            if (mRenderFinishedSemaphores.size() > (size_t)i)
                vkDestroySemaphore(mDevice, mRenderFinishedSemaphores[i], nullptr);
            if (mImageAvailableSemaphores.size() > (size_t)i)
                vkDestroySemaphore(mDevice, mImageAvailableSemaphores[i], nullptr);
            if (mInFlightFences.size() > (size_t)i)
                vkDestroyFence(mDevice, mInFlightFences[i], nullptr);
        }
        if (mCommandPool != VK_NULL_HANDLE)
            vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
        if (mRenderPass != VK_NULL_HANDLE)
            vkDestroyRenderPass(mDevice, mRenderPass, nullptr);
        vkDestroyDevice(mDevice, nullptr);
    }
    if (mSurface != VK_NULL_HANDLE)
        vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
    if (mInstance != VK_NULL_HANDLE)
        vkDestroyInstance(mInstance, nullptr);
}

const char* GfxRenderingAPIVulkan::GetName() {
    return "Vulkan";
}

// ---------------------------------------------------------------------------
// Context creation
// ---------------------------------------------------------------------------

void GfxRenderingAPIVulkan::CreateInstance() {
    // SDL needs the window to report the instance extensions it requires.
    unsigned int extCount = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(mWindow, &extCount, nullptr)) {
        SPDLOG_ERROR("SDL_Vulkan_GetInstanceExtensions(count) failed: {}", SDL_GetError());
        abort();
    }
    std::vector<const char*> extensions(extCount);
    if (!SDL_Vulkan_GetInstanceExtensions(mWindow, &extCount, extensions.data())) {
        SPDLOG_ERROR("SDL_Vulkan_GetInstanceExtensions(list) failed: {}", SDL_GetError());
        abort();
    }
    if (mEnableValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::vector<const char*> layers;
    if (mEnableValidation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Ship of Harkinian (soh3d)";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Fast3D";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;
    ci.enabledExtensionCount = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount = (uint32_t)layers.size();
    ci.ppEnabledLayerNames = layers.data();

    VkResult res = vkCreateInstance(&ci, nullptr, &mInstance);
    if (res != VK_SUCCESS) {
        // Validation layer may be unavailable; retry without it rather than aborting.
        if (mEnableValidation) {
            SPDLOG_WARN("vkCreateInstance with validation failed ({}); retrying without", (int)res);
            mEnableValidation = false;
            ci.enabledLayerCount = 0;
            ci.enabledExtensionCount = extCount; // drop debug_utils
            VK_CHECK(vkCreateInstance(&ci, nullptr, &mInstance));
        } else {
            SPDLOG_ERROR("vkCreateInstance failed: {}", (int)res);
            abort();
        }
    }
    SPDLOG_INFO("Vulkan instance created ({} extensions, validation={})", extCount, mEnableValidation);
}

void GfxRenderingAPIVulkan::CreateSurface() {
    if (!SDL_Vulkan_CreateSurface(mWindow, mInstance, &mSurface)) {
        SPDLOG_ERROR("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
        abort();
    }
}

void GfxRenderingAPIVulkan::PickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(mInstance, &count, nullptr);
    if (count == 0) {
        SPDLOG_ERROR("No Vulkan physical devices found");
        abort();
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(mInstance, &count, devices.data());

    auto findQueues = [&](VkPhysicalDevice dev, uint32_t& gfx, uint32_t& present) -> bool {
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, qprops.data());
        gfx = present = UINT32_MAX;
        for (uint32_t i = 0; i < qcount; i++) {
            if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                if (gfx == UINT32_MAX)
                    gfx = i;
            }
            VkBool32 supportsPresent = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, mSurface, &supportsPresent);
            if (supportsPresent && present == UINT32_MAX)
                present = i;
        }
        return gfx != UINT32_MAX && present != UINT32_MAX;
    };

    // Prefer a discrete GPU that satisfies our queue requirements.
    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    uint32_t chosenGfx = UINT32_MAX, chosenPresent = UINT32_MAX;
    for (int passDiscrete = 1; passDiscrete >= 0 && chosen == VK_NULL_HANDLE; passDiscrete--) {
        for (VkPhysicalDevice dev : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            bool isDiscrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            if (passDiscrete && !isDiscrete)
                continue;
            uint32_t gfx, present;
            if (findQueues(dev, gfx, present)) {
                chosen = dev;
                chosenGfx = gfx;
                chosenPresent = present;
                break;
            }
        }
    }
    if (chosen == VK_NULL_HANDLE) {
        SPDLOG_ERROR("No suitable Vulkan device (graphics+present)");
        abort();
    }
    mPhysicalDevice = chosen;
    mGraphicsQueueFamily = chosenGfx;
    mPresentQueueFamily = chosenPresent;
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &mPhysicalDeviceProps);
    SPDLOG_INFO("Vulkan device: {} (gfx queue {}, present queue {})", mPhysicalDeviceProps.deviceName,
                mGraphicsQueueFamily, mPresentQueueFamily);
}

void GfxRenderingAPIVulkan::CreateLogicalDevice() {
    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    std::vector<uint32_t> families = { mGraphicsQueueFamily };
    if (mPresentQueueFamily != mGraphicsQueueFamily)
        families.push_back(mPresentQueueFamily);
    for (uint32_t fam : families) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = fam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        queueInfos.push_back(qi);
    }

    const char* deviceExtensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy = VK_FALSE;

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount = (uint32_t)queueInfos.size();
    ci.pQueueCreateInfos = queueInfos.data();
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = deviceExtensions;
    ci.pEnabledFeatures = &features;

    VK_CHECK(vkCreateDevice(mPhysicalDevice, &ci, nullptr, &mDevice));
    vkGetDeviceQueue(mDevice, mGraphicsQueueFamily, 0, &mGraphicsQueue);
    vkGetDeviceQueue(mDevice, mPresentQueueFamily, 0, &mPresentQueue);
}

void GfxRenderingAPIVulkan::CreateSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mPhysicalDevice, mSurface, &caps);

    // Surface format: prefer B8G8R8A8_UNORM.
    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &fmtCount, formats.data());
    VkSurfaceFormatKHR chosenFormat = formats[0];
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosenFormat = f;
            break;
        }
    }
    mSwapchainFormat = chosenFormat.format;

    // Present mode: FIFO is always available.
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;

    // Extent.
    VkExtent2D extent;
    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent;
    } else {
        int w = 0, h = 0;
        SDL_Vulkan_GetDrawableSize(mWindow, &w, &h);
        extent.width = std::clamp((uint32_t)w, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    mSwapchainExtent = extent;

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT; // for the frame-dump readback

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = mSurface;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosenFormat.format;
    ci.imageColorSpace = chosenFormat.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = usage;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = presentMode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = VK_NULL_HANDLE;

    uint32_t queueFamilyIndices[] = { mGraphicsQueueFamily, mPresentQueueFamily };
    if (mGraphicsQueueFamily != mPresentQueueFamily) {
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VK_CHECK(vkCreateSwapchainKHR(mDevice, &ci, nullptr, &mSwapchain));

    uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(mDevice, mSwapchain, &actualCount, nullptr);
    mSwapchainImages.resize(actualCount);
    vkGetSwapchainImagesKHR(mDevice, mSwapchain, &actualCount, mSwapchainImages.data());

    mSwapchainImageViews.resize(actualCount);
    for (uint32_t i = 0; i < actualCount; i++) {
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = mSwapchainImages[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = mSwapchainFormat;
        vi.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                          VK_COMPONENT_SWIZZLE_IDENTITY };
        vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(mDevice, &vi, nullptr, &mSwapchainImageViews[i]));
    }
    SPDLOG_INFO("Vulkan swapchain: {}x{}, {} images, format {}", extent.width, extent.height, actualCount,
                (int)mSwapchainFormat);
}

void GfxRenderingAPIVulkan::CreateRenderPass() {
    VkAttachmentDescription color{};
    color.format = mSwapchainFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = 1;
    ci.pAttachments = &color;
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;

    VK_CHECK(vkCreateRenderPass(mDevice, &ci, nullptr, &mRenderPass));
}

void GfxRenderingAPIVulkan::CreateFramebuffers() {
    mSwapchainFramebuffers.resize(mSwapchainImageViews.size());
    for (size_t i = 0; i < mSwapchainImageViews.size(); i++) {
        VkImageView attachments[] = { mSwapchainImageViews[i] };
        VkFramebufferCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass = mRenderPass;
        fi.attachmentCount = 1;
        fi.pAttachments = attachments;
        fi.width = mSwapchainExtent.width;
        fi.height = mSwapchainExtent.height;
        fi.layers = 1;
        VK_CHECK(vkCreateFramebuffer(mDevice, &fi, nullptr, &mSwapchainFramebuffers[i]));
    }
}

void GfxRenderingAPIVulkan::CreateCommandResources() {
    VkCommandPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = mGraphicsQueueFamily;
    VK_CHECK(vkCreateCommandPool(mDevice, &pi, nullptr, &mCommandPool));

    mCommandBuffers.resize(kMaxFramesInFlight);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = mCommandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = kMaxFramesInFlight;
    VK_CHECK(vkAllocateCommandBuffers(mDevice, &ai, mCommandBuffers.data()));
}

void GfxRenderingAPIVulkan::CreateSyncObjects() {
    mImageAvailableSemaphores.resize(kMaxFramesInFlight);
    mRenderFinishedSemaphores.resize(kMaxFramesInFlight);
    mInFlightFences.resize(kMaxFramesInFlight);

    VkSemaphoreCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < kMaxFramesInFlight; i++) {
        VK_CHECK(vkCreateSemaphore(mDevice, &si, nullptr, &mImageAvailableSemaphores[i]));
        VK_CHECK(vkCreateSemaphore(mDevice, &si, nullptr, &mRenderFinishedSemaphores[i]));
        VK_CHECK(vkCreateFence(mDevice, &fi, nullptr, &mInFlightFences[i]));
    }
}

void GfxRenderingAPIVulkan::DestroySwapchain() {
    for (VkFramebuffer fb : mSwapchainFramebuffers)
        vkDestroyFramebuffer(mDevice, fb, nullptr);
    mSwapchainFramebuffers.clear();
    for (VkImageView iv : mSwapchainImageViews)
        vkDestroyImageView(mDevice, iv, nullptr);
    mSwapchainImageViews.clear();
    if (mSwapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(mDevice, mSwapchain, nullptr);
        mSwapchain = VK_NULL_HANDLE;
    }
}

void GfxRenderingAPIVulkan::RecreateSwapchain() {
    vkDeviceWaitIdle(mDevice);
    DestroySwapchain();
    CreateSwapchain();
    CreateFramebuffers();
}

void GfxRenderingAPIVulkan::Init() {
    mWindow = mWindowBackend ? mWindowBackend->GetSdlWindow() : nullptr;
    if (mWindow == nullptr) {
        SPDLOG_ERROR("Vulkan backend: SDL window is null at Init()");
        abort();
    }
    CreateInstance();
    CreateSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapchain();
    CreateRenderPass();
    CreateFramebuffers();
    CreateCommandResources();
    CreateSyncObjects();
    SPDLOG_INFO("Vulkan backend initialized (Milestone 1: clear-only)");
}

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

void GfxRenderingAPIVulkan::StartFrame() {
    // Idempotent within a displayed frame: the interpreter calls StartFrame twice
    // (Interpreter::StartFrame and Interpreter::Run). Only the first acquires.
    if (mFrameAcquired) {
        return;
    }

    vkWaitForFences(mDevice, 1, &mInFlightFences[mCurrentFrame], VK_TRUE, UINT64_MAX);

    VkResult acq = vkAcquireNextImageKHR(mDevice, mSwapchain, UINT64_MAX,
                                         mImageAvailableSemaphores[mCurrentFrame], VK_NULL_HANDLE, &mImageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        RecreateSwapchain();
        return; // skip this frame
    } else if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
        SPDLOG_ERROR("vkAcquireNextImageKHR failed: {}", (int)acq);
        abort();
    }

    vkResetFences(mDevice, 1, &mInFlightFences[mCurrentFrame]);

    VkCommandBuffer cmd = mCommandBuffers[mCurrentFrame];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    VkClearValue clear{};
    clear.color = { { mClearColor[0], mClearColor[1], mClearColor[2], mClearColor[3] } };

    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = mRenderPass;
    rp.framebuffer = mSwapchainFramebuffers[mImageIndex];
    rp.renderArea.offset = { 0, 0 };
    rp.renderArea.extent = mSwapchainExtent;
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    mFrameAcquired = true;
}

void GfxRenderingAPIVulkan::EndFrame() {
    if (!mFrameAcquired) {
        return;
    }
    VkCommandBuffer cmd = mCommandBuffers[mCurrentFrame];
    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));
}

void GfxRenderingAPIVulkan::FinishRender() {
    if (!mFrameAcquired) {
        return;
    }
    VkCommandBuffer cmd = mCommandBuffers[mCurrentFrame];

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkSemaphore waitSems[] = { mImageAvailableSemaphores[mCurrentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = waitSems;
    submit.pWaitDstStageMask = waitStages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VkSemaphore signalSems[] = { mRenderFinishedSemaphores[mCurrentFrame] };
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = signalSems;

    VK_CHECK(vkQueueSubmit(mGraphicsQueue, 1, &submit, mInFlightFences[mCurrentFrame]));

    // On-demand / scripted frame dump (verification). Reads the rendered swapchain
    // image after waiting for the submit to finish. Slow, only when triggered.
    MaybeDumpFrame();

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = signalSems;
    VkSwapchainKHR swapchains[] = { mSwapchain };
    present.swapchainCount = 1;
    present.pSwapchains = swapchains;
    present.pImageIndices = &mImageIndex;

    VkResult pres = vkQueuePresentKHR(mPresentQueue, &present);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
        RecreateSwapchain();
    } else if (pres != VK_SUCCESS) {
        SPDLOG_ERROR("vkQueuePresentKHR failed: {}", (int)pres);
        abort();
    }

    mCurrentFrame = (mCurrentFrame + 1) % kMaxFramesInFlight;
    mFrameAcquired = false;
}

// ---------------------------------------------------------------------------
// Frame dump (PPM readback of the swapchain image)
// ---------------------------------------------------------------------------

void GfxRenderingAPIVulkan::MaybeDumpFrame() {
    const char* path = nullptr;

    // 1) Scripted single-frame dump: SOH_FRAMEDUMP=<path>, at SOH_FRAMEDUMP_FRAME.
    static const char* envDump = getenv("SOH_FRAMEDUMP");
    static long frame = 0;
    static long targetFrame = getenv("SOH_FRAMEDUMP_FRAME") ? atol(getenv("SOH_FRAMEDUMP_FRAME")) : 300;
    bool exitAfter = false;
    if (envDump != nullptr) {
        ++frame;
        if (frame == targetFrame) {
            path = envDump;
            exitAfter = true;
        }
    }

    // 2) REPL on-demand dump (does not exit). Takes priority if both fire.
    if (gSoh3dDumpPending) {
        path = gSoh3dDumpPath;
        exitAfter = false;
    }

    if (path == nullptr) {
        return;
    }

    // Ensure the render submit for this frame has completed before reading.
    vkWaitForFences(mDevice, 1, &mInFlightFences[mCurrentFrame], VK_TRUE, UINT64_MAX);
    WriteSwapchainPpm(path);
    gSoh3dDumpPending = 0;
    if (exitAfter) {
        exit(0);
    }
}

void GfxRenderingAPIVulkan::WriteSwapchainPpm(const char* path) {
    const uint32_t w = mSwapchainExtent.width;
    const uint32_t h = mSwapchainExtent.height;
    const VkDeviceSize size = (VkDeviceSize)w * h * 4;

    // Host-visible staging buffer.
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(mDevice, &bci, nullptr, &buffer));

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(mDevice, buffer, &memReq);
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memProps);
    uint32_t memType = UINT32_MAX;
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReq.memoryTypeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & want) == want) {
            memType = i;
            break;
        }
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = memReq.size;
    mai.memoryTypeIndex = memType;
    VK_CHECK(vkAllocateMemory(mDevice, &mai, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(mDevice, buffer, memory, 0));

    // One-shot command buffer: PRESENT_SRC -> TRANSFER_SRC, copy, -> PRESENT_SRC.
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = mCommandPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(mDevice, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &cbi));

    VkImage srcImage = mSwapchainImages[mImageIndex];
    auto barrier = [&](VkImageLayout oldL, VkImageLayout newL, VkAccessFlags srcA, VkAccessFlags dstA,
                       VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = oldL;
        b.newLayout = newL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = srcImage;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask = srcA;
        b.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    barrier(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { w, h, 1 };
    vkCmdCopyImageToBuffer(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_MEMORY_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VkFence fence;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(mDevice, &fci, nullptr, &fence));
    VK_CHECK(vkQueueSubmit(mGraphicsQueue, 1, &submit, fence));
    vkWaitForFences(mDevice, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(mDevice, fence, nullptr);
    vkFreeCommandBuffers(mDevice, mCommandPool, 1, &cmd);

    // Map and write PPM (top-down).
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(mDevice, memory, 0, size, 0, &mapped));
    const uint8_t* px = static_cast<const uint8_t*>(mapped);
    bool bgra = (mSwapchainFormat == VK_FORMAT_B8G8R8A8_UNORM || mSwapchainFormat == VK_FORMAT_B8G8R8A8_SRGB);
    FILE* f = fopen(path, "wb");
    if (f) {
        fprintf(f, "P6\n%u %u\n255\n", w, h);
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                const uint8_t* p = &px[((size_t)y * w + x) * 4];
                uint8_t rgb[3];
                if (bgra) {
                    rgb[0] = p[2];
                    rgb[1] = p[1];
                    rgb[2] = p[0];
                } else {
                    rgb[0] = p[0];
                    rgb[1] = p[1];
                    rgb[2] = p[2];
                }
                fwrite(rgb, 1, 3, f);
            }
        }
        fclose(f);
        SPDLOG_INFO("Vulkan frame dump written: {} ({}x{})", path, w, h);
    } else {
        SPDLOG_ERROR("Vulkan frame dump: could not open {}", path);
    }
    vkUnmapMemory(mDevice, memory);
    vkDestroyBuffer(mDevice, buffer, nullptr);
    vkFreeMemory(mDevice, memory, nullptr);
}

void GfxRenderingAPIVulkan::OnResize() {
    // Swapchain is recreated lazily on OUT_OF_DATE/SUBOPTIMAL during acquire/present.
}

// ---------------------------------------------------------------------------
// Shader records (no GPU pipeline in M1; just feature bookkeeping)
// ---------------------------------------------------------------------------

void GfxRenderingAPIVulkan::ClearShaderCache() {
    mShaderProgramPool.clear();
}

ShaderProgram* GfxRenderingAPIVulkan::CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) {
    CCFeatures cc{};
    gfx_cc_get_features(shaderId0, shaderId1, &cc);
    ShaderProgramVulkan& prg = mShaderProgramPool[std::make_pair(shaderId0, shaderId1)];
    prg.numInputs = cc.numInputs;
    prg.usedTextures[0] = cc.usedTextures[0];
    prg.usedTextures[1] = cc.usedTextures[1];
    return reinterpret_cast<ShaderProgram*>(&prg);
}

ShaderProgram* GfxRenderingAPIVulkan::LookupShader(uint64_t shaderId0, uint64_t shaderId1) {
    auto it = mShaderProgramPool.find(std::make_pair(shaderId0, shaderId1));
    return it == mShaderProgramPool.end() ? nullptr : reinterpret_cast<ShaderProgram*>(&it->second);
}

void GfxRenderingAPIVulkan::ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) {
    auto* vk = reinterpret_cast<ShaderProgramVulkan*>(prg);
    if (vk == nullptr) {
        *numInputs = 0;
        usedTextures[0] = usedTextures[1] = false;
        return;
    }
    *numInputs = vk->numInputs;
    usedTextures[0] = vk->usedTextures[0];
    usedTextures[1] = vk->usedTextures[1];
}

void GfxRenderingAPIVulkan::LoadShader(ShaderProgram*) {
}
void GfxRenderingAPIVulkan::UnloadShader(ShaderProgram*) {
}

// ---------------------------------------------------------------------------
// Inert entry points (implemented in later milestones)
// ---------------------------------------------------------------------------

int GfxRenderingAPIVulkan::GetMaxTextureSize() {
    return mPhysicalDeviceProps.limits.maxImageDimension2D ? (int)mPhysicalDeviceProps.limits.maxImageDimension2D
                                                           : 4096;
}
GfxClipParameters GfxRenderingAPIVulkan::GetClipParameters() {
    // Vulkan clip space: z in [0,1], Y points down (invert).
    return { true, true };
}
uint32_t GfxRenderingAPIVulkan::NewTexture() {
    return mNextTextureId++;
}
void GfxRenderingAPIVulkan::SelectTexture(int, uint32_t) {
}
void GfxRenderingAPIVulkan::UploadTexture(const uint8_t*, uint32_t, uint32_t) {
}
void GfxRenderingAPIVulkan::SetSamplerParameters(int, bool, uint32_t, uint32_t) {
}
void GfxRenderingAPIVulkan::SetDepthTestAndMask(bool depthTest, bool zUpd) {
    mCurrentDepthTest = depthTest;
    mCurrentDepthMask = zUpd;
}
void GfxRenderingAPIVulkan::SetZmodeDecal(bool decal) {
    mCurrentZmodeDecal = decal;
}
void GfxRenderingAPIVulkan::SetViewport(int, int, int, int) {
}
void GfxRenderingAPIVulkan::SetScissor(int, int, int, int) {
}
void GfxRenderingAPIVulkan::SetUseAlpha(bool) {
}
void GfxRenderingAPIVulkan::DrawTriangles(float[], size_t, size_t) {
}
int GfxRenderingAPIVulkan::CreateFramebuffer() {
    return mFramebufferCount++;
}
void GfxRenderingAPIVulkan::UpdateFramebufferParameters(int, uint32_t, uint32_t, uint32_t, bool, bool, bool, bool) {
}
void GfxRenderingAPIVulkan::StartDrawToFramebuffer(int, float) {
}
void GfxRenderingAPIVulkan::CopyFramebuffer(int, int, int, int, int, int, int, int, int, int) {
}
void GfxRenderingAPIVulkan::ClearFramebuffer(bool, bool) {
}
void GfxRenderingAPIVulkan::ReadFramebufferToCPU(int, uint32_t, uint32_t, uint16_t*) {
}
void GfxRenderingAPIVulkan::ResolveMSAAColorBuffer(int, int) {
}
std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
GfxRenderingAPIVulkan::GetPixelDepth(int, const std::set<std::pair<float, float>>& coordinates) {
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res;
    for (const auto& c : coordinates)
        res.emplace(c, 0);
    return res;
}
void* GfxRenderingAPIVulkan::GetFramebufferTextureId(int) {
    return nullptr;
}
void GfxRenderingAPIVulkan::SelectTextureFb(int) {
}
void GfxRenderingAPIVulkan::DeleteTexture(uint32_t) {
}
void GfxRenderingAPIVulkan::SetTextureFilter(FilteringMode mode) {
    mCurrentFilterMode = mode;
}
FilteringMode GfxRenderingAPIVulkan::GetTextureFilter() {
    return mCurrentFilterMode;
}
void GfxRenderingAPIVulkan::SetSrgbMode() {
    mSrgbMode = true;
}
ImTextureID GfxRenderingAPIVulkan::GetTextureById(int id) {
    return reinterpret_cast<ImTextureID>((uintptr_t)id);
}
void GfxRenderingAPIVulkan::SetCurrentPrimDepth(float depth) {
    mCurrentPrimDepth = depth;
}

} // namespace Fast

#endif // ENABLE_VULKAN
