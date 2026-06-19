// SoH3D Vulkan render pass. See include/fast/soh3d_vk.h for the role split with soh3d_gl.cpp.
//
// Renders the collected OoT3D draw items (textured / GPU-skinned / half-Lambert-lit, per-group
// blend + depth-write + alpha-test + decal depth-bias + mesh_id visibility) into the Fast3D Vulkan
// backend's current command buffer + render pass, so the 3DS content interleaves depth-correctly
// with the N64 geometry. Mirrors soh3d_gl.cpp's drawOne; the per-item pose interpolation is done by
// the shared SoH3D_GL_RenderPass which calls SoH3D_Vk_DrawModel per item.
//
// Dynamic sun-shadows + screen-space AO (the GL pass's extra offscreen passes) are NOT ported here
// yet — this is the core content. They are a self-contained follow-up.
#ifdef ENABLE_VULKAN

#include "fast/soh3d_vk.h"
#include "fast/backends/gfx_vulkan.h"

#include <vulkan/vulkan.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <vector>
#include <map>
#include <unordered_map>
#include <array>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <cmath>
#include <mutex>

using Fast::SoH3DVkContext;

// World-space sun direction, owned by soh3d_gl.cpp (set per frame by soh3d.c). C linkage.
extern "C" float gSoH3dLightDirWorld[3];
// Backface culling (shared toggle with the GL backend; see soh3d_gl.cpp). -1 = resolve from
// env SOH3D_FACECULL (default ON). gSoH3dFaceCullFlip flips the front-face winding convention.
extern "C" int gSoH3dFaceCull;
extern "C" int gSoH3dFaceCullFlip;
static int vkFaceCullOn() {
    if (gSoH3dFaceCull < 0) {
        const char* e = getenv("SOH3D_FACECULL");
        gSoH3dFaceCull = (e && e[0] == '0') ? 0 : 1; // default ON
    }
    return gSoH3dFaceCull;
}

namespace {

// ---- GLSL -> SPIR-V (glslang; the same toolchain the backend uses) ----
std::once_flag g_glslOnce;
bool CompileGlsl(EShLanguage stage, const char* src, std::vector<uint32_t>& spv) {
    std::call_once(g_glslOnce, []() { glslang::InitializeProcess(); });
    glslang::TShader shader(stage);
    shader.setStrings(&src, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);
    EShMessages msg = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
    if (!shader.parse(GetDefaultResources(), 450, false, msg)) {
        fprintf(stderr, "[SoH3D_VK] shader parse failed: %s\n", shader.getInfoLog());
        return false;
    }
    glslang::TProgram prog;
    prog.addShader(&shader);
    if (!prog.link(msg)) {
        fprintf(stderr, "[SoH3D_VK] shader link failed: %s\n", prog.getInfoLog());
        return false;
    }
    glslang::SpvOptions opt;
    opt.disableOptimizer = true;
    glslang::GlslangToSpv(*prog.getIntermediate(stage), spv, &opt);
    return !spv.empty();
}

const char* kVert = R"(#version 450
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aUv;
layout(location=3) in vec4 aBoneId;
layout(location=4) in vec4 aBoneW;
layout(location=5) in vec4 aColor;
layout(location=0) out vec2 vUv;
layout(location=1) out vec4 vColor;
layout(location=2) out vec3 vNrmView;
layout(binding=0, std140) uniform UBO {
    mat4 uMP;
    mat4 uMV;
    mat4 uBones[32];
    vec4 uLightDir;  // xyz: world-space sun dir; w: 1 = skybox dome (pin to far plane)
    vec4 uParams;    // x=invertY(+1/-1) y=lit z=alphaRef w=depthOffset
    vec4 uTintSkin;  // xyz=tint w=skin(0/1)
    vec4 uExtra;     // x=per-draw alpha (1=opaque) y=texcoord scroll U z=scroll V (cloud drift, #28b)
} ubo;
void main() {
    vColor = aColor;
    vec3 sp, nM;
    if (ubo.uTintSkin.w > 0.5) {
        vec4 acc = vec4(0.0); nM = vec3(0.0);
        for (int i = 0; i < 4; i++) {
            acc += aBoneW[i] * (ubo.uBones[int(aBoneId[i])] * vec4(aPos, 1.0));
            nM  += aBoneW[i] * (mat3(ubo.uBones[int(aBoneId[i])]) * aNrm);
        }
        sp = acc.xyz;
    } else { sp = aPos; nM = aNrm; }
    vec4 c = ubo.uMP * vec4(sp, 1.0);
    // uParams.x carries the backend's clip invertY (interpreter passes GetClipParameters().invertY:
    // -1 on Vulkan, +1 on GL). That single negate IS the Vulkan Y-down flip and matches exactly what
    // the interpreter applies to N64 vertices (interpreter.cpp: y = -y). Do NOT negate again.
    c.y *= ubo.uParams.x;
    c.z = (c.z + c.w) * 0.5;  // GL clip z [-1,1] -> Vulkan [0,1]
    if (ubo.uLightDir.w > 0.5) c.z = c.w; // skybox: pin to far plane (Vulkan far = z/w = 1)
    gl_Position = c;
    vNrmView = mat3(ubo.uMV) * nM; // world-space normal (uMV is model->world; see soh3d_gl.cpp)
    // CMB/PICA UVs are top-origin. Texture SAMPLING maps v=0 -> data row 0 identically in GL and
    // Vulkan (the bottom-left/top-left API difference is framebuffer-only, NOT texture data), so the
    // same 1-v flip GL uses is required here too. (Visible only on detailed texels - face/emblem -
    // not on near-uniform cloth, which is why it looked fine at first.)
    vUv = vec2(aUv.x + ubo.uExtra.y, 1.0 - aUv.y + ubo.uExtra.z); // + per-draw cloud-band drift (#28b)
}
)";

const char* kFrag = R"(#version 450
layout(location=0) in vec2 vUv;
layout(location=1) in vec4 vColor;
layout(location=2) in vec3 vNrmView;
layout(location=0) out vec4 frag;
layout(binding=0, std140) uniform UBO {
    mat4 uMP;
    mat4 uMV;
    mat4 uBones[32];
    vec4 uLightDir;
    vec4 uParams;
    vec4 uTintSkin;
    vec4 uExtra;
} ubo;
layout(binding=1) uniform sampler2D uTex;
void main() {
    vec4 t = texture(uTex, vUv);
    if (t.a < ubo.uParams.z) discard;
    gl_FragDepth = gl_FragCoord.z + ubo.uParams.w; // decal depth bias (polygon offset)
    vec3 shade = ubo.uTintSkin.xyz;
    if (ubo.uParams.y > 0.5) { // half-Lambert form term for characters/props
        float hl = dot(normalize(vNrmView), normalize(ubo.uLightDir.xyz)) * 0.5 + 0.5;
        shade = ubo.uTintSkin.xyz * (0.55 + 0.45 * hl);
    }
    frag = vec4(t.rgb * vColor.rgb * shade, t.a * vColor.a * ubo.uExtra.x); // uExtra.x = per-draw alpha
}
)";

// std140 UBO layout matching the shader block (2224 bytes).
struct VkUbo {
    float uMP[16];
    float uMV[16];
    float uBones[32 * 16];
    float uLightDir[4];
    float uParams[4];
    float uTintSkin[4];
    float uExtra[4]; // x = per-draw alpha (1 = opaque); y/z = texcoord scroll U/V (cloud drift, #28b)
};

struct VkTex {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

struct VkGroup {
    uint32_t first = 0, count = 0;
    int texIndex = -1;
    int alphaTest = 0;
    float alphaRef = 0.0f;
    unsigned wrapS = 0x2901, wrapT = 0x2901;
    int blendEnable = 0;
    unsigned bSrcRGB = 0x0302, bDstRGB = 0x0303, bEqRGB = 0x8006;
    unsigned bSrcA = 1, bDstA = 0, bEqA = 0x8006;
    float blendColor[4] = { 0, 0, 0, 1 };
    int depthWrite = 1;
    float polygonOffset = 0.0f;
    int cull = 0;
    int faceCull = 0; // 1 = cull back face (CMB cull byte 1); 0 = double-sided
    int meshId = -1;
};

struct VkModel {
    bool uploaded = false, failed = false;
    VkBuffer vbo = VK_NULL_HANDLE;
    VkDeviceMemory vboMem = VK_NULL_HANDLE;
    std::vector<VkGroup> groups;
    std::vector<VkTex> textures;
};

struct Ring {
    VkBuffer ubo = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize capacity = 0;
    VkDeviceSize offset = 0;
    VkDescriptorPool pool = VK_NULL_HANDLE;
};

// ---- module state ----
SoH3DModelProvider g_provider = nullptr;
std::unordered_map<int, VkModel> g_models;

SoH3DVkContext g_ctx{};
bool g_ctxValid = false;

bool g_resReady = false;
VkDevice g_device = VK_NULL_HANDLE;
VkPhysicalDevice g_phys = VK_NULL_HANDLE;
VkDescriptorSetLayout g_setLayout = VK_NULL_HANDLE;
VkPipelineLayout g_pipeLayout = VK_NULL_HANDLE;
VkShaderModule g_vsMod = VK_NULL_HANDLE, g_fsMod = VK_NULL_HANDLE;
VkRenderPass g_renderPass = VK_NULL_HANDLE;
std::map<std::array<uint32_t, 7>, VkPipeline> g_pipelines; // key: flags + 6 blend params
std::map<uint32_t, VkSampler> g_samplers;                  // key: (wrapS<<16)|wrapT
VkTex g_dummyTex{};
VkSampler g_dummySampler = VK_NULL_HANDLE;
VkDeviceSize g_uboStride = 0;
std::vector<Ring> g_rings; // per frame-in-flight

constexpr uint32_t kMaxGroupsPerFrame = 4096;

uint32_t findMemType(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return 0;
}

void makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf,
                VkDeviceMemory& mem, void** mappedOut) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(g_device, &bi, nullptr, &buf);
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_device, buf, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits, props);
    vkAllocateMemory(g_device, &ai, nullptr, &mem);
    vkBindBufferMemory(g_device, buf, mem, 0);
    if (mappedOut)
        vkMapMemory(g_device, mem, 0, size, 0, mappedOut);
}

// Run a one-shot transfer command and wait. (Models upload once; not perf-critical.)
template <typename F> void oneShot(F record) {
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = g_ctx.commandPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(g_device, &cai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    record(cmd);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkFence fence;
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(g_device, &fi, nullptr, &fence);
    vkQueueSubmit(g_ctx.graphicsQueue, 1, &si, fence);
    vkWaitForFences(g_device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(g_device, fence, nullptr);
    vkFreeCommandBuffers(g_device, g_ctx.commandPool, 1, &cmd);
}

void uploadTexture(VkTex& t, int w, int h, const unsigned char* rgba) {
    if (w <= 0 || h <= 0)
        w = h = 1;
    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = { (uint32_t)w, (uint32_t)h, 1 };
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(g_device, &ii, nullptr, &t.image);
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g_device, t.image, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(g_device, &ai, nullptr, &t.mem);
    vkBindImageMemory(g_device, t.image, t.mem, 0);

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = t.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(g_device, &vi, nullptr, &t.view);

    const VkDeviceSize size = (VkDeviceSize)w * h * 4;
    static const unsigned char white[4] = { 255, 255, 255, 255 };
    const unsigned char* src = rgba ? rgba : white;
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    void* mapped = nullptr;
    makeBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem,
               &mapped);
    memcpy(mapped, src, rgba ? size : 4);
    vkUnmapMemory(g_device, stagingMem);

    oneShot([&](VkCommandBuffer cmd) {
        auto barrier = [&](VkImageLayout o, VkImageLayout n, VkAccessFlags sa, VkAccessFlags da,
                           VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = o;
            b.newLayout = n;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = t.image;
            b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            b.srcAccessMask = sa;
            b.dstAccessMask = da;
            vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
        };
        barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };
        vkCmdCopyBufferToImage(cmd, staging, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    });
    vkDestroyBuffer(g_device, staging, nullptr);
    vkFreeMemory(g_device, stagingMem, nullptr);
}

VkSamplerAddressMode wrapMode(unsigned glWrap) {
    switch (glWrap) {
        case 0x2900: // GL_CLAMP
        case 0x812F: // GL_CLAMP_TO_EDGE
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 0x8370: // GL_MIRRORED_REPEAT
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        default: // 0x2901 GL_REPEAT
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

VkSampler getSampler(unsigned wrapS, unsigned wrapT) {
    uint32_t key = (wrapS << 16) | (wrapT & 0xFFFF);
    auto it = g_samplers.find(key);
    if (it != g_samplers.end())
        return it->second;
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = wrapMode(wrapS);
    si.addressModeV = wrapMode(wrapT);
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.maxLod = 0.0f;
    VkSampler s;
    vkCreateSampler(g_device, &si, nullptr, &s);
    g_samplers[key] = s;
    return s;
}

VkBlendFactor mapFactor(unsigned f) {
    switch (f) {
        case 0: return VK_BLEND_FACTOR_ZERO;
        case 1: return VK_BLEND_FACTOR_ONE;
        case 0x300: return VK_BLEND_FACTOR_SRC_COLOR;
        case 0x301: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 0x302: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 0x303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 0x304: return VK_BLEND_FACTOR_DST_ALPHA;
        case 0x305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 0x306: return VK_BLEND_FACTOR_DST_COLOR;
        case 0x307: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 0x308: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x8001: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x8002: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 0x8003: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}
VkBlendOp mapEq(unsigned e) {
    switch (e) {
        case 0x8006: return VK_BLEND_OP_ADD;
        case 0x800A: return VK_BLEND_OP_SUBTRACT;
        case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 0x8007: return VK_BLEND_OP_MIN;
        case 0x8008: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}

VkShaderModule makeModule(const std::vector<uint32_t>& spv) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spv.size() * sizeof(uint32_t);
    ci.pCode = spv.data();
    VkShaderModule m = VK_NULL_HANDLE;
    vkCreateShaderModule(g_device, &ci, nullptr, &m);
    return m;
}

bool ensureResources(const SoH3DVkContext& ctx) {
    if (g_resReady)
        return true;
    g_device = ctx.device;
    g_phys = ctx.physicalDevice;
    g_renderPass = ctx.renderPass;

    std::vector<uint32_t> vsSpv, fsSpv;
    if (!CompileGlsl(EShLangVertex, kVert, vsSpv) || !CompileGlsl(EShLangFragment, kFrag, fsSpv))
        return false;
    g_vsMod = makeModule(vsSpv);
    g_fsMod = makeModule(fsSpv);

    VkDescriptorSetLayoutBinding binds[2]{};
    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 2;
    dli.pBindings = binds;
    vkCreateDescriptorSetLayout(g_device, &dli, nullptr, &g_setLayout);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_setLayout;
    vkCreatePipelineLayout(g_device, &pli, nullptr, &g_pipeLayout);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_phys, &props);
    VkDeviceSize align = props.limits.minUniformBufferOffsetAlignment;
    if (align == 0)
        align = 1;
    g_uboStride = ((sizeof(VkUbo) + align - 1) / align) * align;

    g_rings.resize(ctx.framesInFlight);
    for (uint32_t i = 0; i < ctx.framesInFlight; i++) {
        Ring& r = g_rings[i];
        r.capacity = g_uboStride * kMaxGroupsPerFrame;
        makeBuffer(r.capacity, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, r.ubo, r.mem,
                   &r.mapped);
        VkDescriptorPoolSize ps[2]{};
        ps[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ps[0].descriptorCount = kMaxGroupsPerFrame;
        ps[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[1].descriptorCount = kMaxGroupsPerFrame;
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = kMaxGroupsPerFrame;
        dpi.poolSizeCount = 2;
        dpi.pPoolSizes = ps;
        vkCreateDescriptorPool(g_device, &dpi, nullptr, &r.pool);
    }

    uploadTexture(g_dummyTex, 1, 1, nullptr); // 1x1 white for untextured groups
    g_dummySampler = getSampler(0x2901, 0x2901);
    g_resReady = true;
    fprintf(stderr, "[SoH3D_VK] resources ready (ubo stride %llu)\n", (unsigned long long)g_uboStride);
    return true;
}

VkPipeline getPipeline(const VkGroup& g, int frontCW) {
    // Backface cull is baked into the pipeline (cullMode/frontFace are not dynamic here), so the
    // cull intent + the winding (which flips with invertY, carried in frontCW) join the key.
    bool doCull = g.faceCull && vkFaceCullOn();
    std::array<uint32_t, 7> key = { (uint32_t)((g.blendEnable ? 1u : 0u) | (g.depthWrite ? 2u : 0u) |
                                               (doCull ? 4u : 0u) | (doCull && frontCW ? 8u : 0u)),
                                    g.bSrcRGB, g.bDstRGB, g.bEqRGB, g.bSrcA, g.bDstA, g.bEqA };
    auto it = g_pipelines.find(key);
    if (it != g_pipelines.end())
        return it->second;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = g_vsMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = g_fsMod;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(SoH3DGlVtx);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrs[6]{};
    attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(SoH3DGlVtx, pos) };
    attrs[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(SoH3DGlVtx, nrm) };
    attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT, (uint32_t)offsetof(SoH3DGlVtx, uv) };
    attrs[3] = { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(SoH3DGlVtx, boneIds) };
    attrs[4] = { 4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(SoH3DGlVtx, weights) };
    attrs[5] = { 5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(SoH3DGlVtx, color) };
    VkPipelineVertexInputStateCreateInfo vin{};
    vin.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vin.vertexBindingDescriptionCount = 1;
    vin.pVertexBindingDescriptions = &binding;
    vin.vertexAttributeDescriptionCount = 6;
    vin.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.depthClampEnable = VK_TRUE; // device feature is on; matches the Fast3D backend
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    // Honor the CMB material cull byte (1 = cull back). The asset winds front faces CCW from the
    // geometric normal; the vertex shader negates clip.y when invertY, flipping window winding ->
    // frontCW carries that (plus the gSoH3dFaceCullFlip convention toggle). Double-sided groups
    // (faceCull 0 / cull disabled) keep VK_CULL_MODE_NONE.
    rs.cullMode = doCull ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    rs.frontFace = frontCW ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = g.depthWrite ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cba.blendEnable = g.blendEnable ? VK_TRUE : VK_FALSE;
    cba.srcColorBlendFactor = mapFactor(g.bSrcRGB);
    cba.dstColorBlendFactor = mapFactor(g.bDstRGB);
    cba.colorBlendOp = mapEq(g.bEqRGB);
    cba.srcAlphaBlendFactor = mapFactor(g.bSrcA);
    cba.dstAlphaBlendFactor = mapFactor(g.bDstA);
    cba.alphaBlendOp = mapEq(g.bEqA);
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS };
    VkPipelineDynamicStateCreateInfo dynS{};
    dynS.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynS.dynamicStateCount = 3;
    dynS.pDynamicStates = dyn;

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vin;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &dynS;
    pci.layout = g_pipeLayout;
    pci.renderPass = g_renderPass;
    pci.subpass = 0;
    VkPipeline pipe = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(g_device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe);
    g_pipelines[key] = pipe;
    return pipe;
}

VkModel* ensureUploaded(int modelId) {
    VkModel& m = g_models[modelId];
    if (m.uploaded)
        return &m;
    if (m.failed)
        return nullptr;
    const SoH3DGlGroup* groups = nullptr;
    const SoH3DGlTex* texs = nullptr;
    int groupCount = 0, texCount = 0;
    if (!g_provider || !g_provider(modelId, &groups, &groupCount, &texs, &texCount) || groupCount <= 0) {
        fprintf(stderr, "[SoH3D_VK] model %d unavailable from provider\n", modelId);
        m.failed = true;
        return nullptr;
    }

    std::vector<SoH3DGlVtx> all;
    for (int i = 0; i < groupCount; i++) {
        VkGroup g;
        g.first = (uint32_t)all.size();
        g.count = (uint32_t)groups[i].vertCount;
        g.texIndex = groups[i].texIndex;
        g.alphaTest = groups[i].alphaTest;
        g.alphaRef = groups[i].alphaRef;
        g.wrapS = groups[i].wrapS;
        g.wrapT = groups[i].wrapT;
        g.blendEnable = groups[i].blendEnable;
        g.bSrcRGB = groups[i].blendSrcRGB;
        g.bDstRGB = groups[i].blendDstRGB;
        g.bEqRGB = groups[i].blendEqRGB;
        g.bSrcA = groups[i].blendSrcA;
        g.bDstA = groups[i].blendDstA;
        g.bEqA = groups[i].blendEqA;
        g.depthWrite = groups[i].depthWrite;
        g.polygonOffset = groups[i].polygonOffset;
        g.cull = groups[i].cull;
        g.faceCull = groups[i].faceCull;
        g.meshId = groups[i].meshId;
        for (int k = 0; k < 4; k++)
            g.blendColor[k] = groups[i].blendColor[k];
        all.insert(all.end(), groups[i].verts, groups[i].verts + groups[i].vertCount);
        m.groups.push_back(g);
    }

    // Device-local vertex buffer via a staging copy.
    const VkDeviceSize vbBytes = all.size() * sizeof(SoH3DGlVtx);
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    void* mapped = nullptr;
    makeBuffer(vbBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem,
               &mapped);
    memcpy(mapped, all.data(), vbBytes);
    vkUnmapMemory(g_device, stagingMem);
    makeBuffer(vbBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.vbo, m.vboMem, nullptr);
    oneShot([&](VkCommandBuffer cmd) {
        VkBufferCopy c{};
        c.size = vbBytes;
        vkCmdCopyBuffer(cmd, staging, m.vbo, 1, &c);
    });
    vkDestroyBuffer(g_device, staging, nullptr);
    vkFreeMemory(g_device, stagingMem, nullptr);

    for (int i = 0; i < texCount; i++) {
        VkTex t;
        uploadTexture(t, texs[i].w, texs[i].h, texs[i].rgba);
        m.textures.push_back(t);
    }
    m.uploaded = true;
    fprintf(stderr, "[SoH3D_VK] uploaded model %d: %d groups, %d textures, %zu verts\n", modelId, groupCount,
            texCount, all.size());
    return &m;
}

} // namespace

extern "C" int SoH3D_Vk_Active(void) {
    return Fast::g_activeVulkanApi != nullptr ? 1 : 0;
}

extern "C" void SoH3D_Vk_SetProvider(SoH3DModelProvider fn) {
    g_provider = fn;
}

// Deferred model-cache eviction (mirror of the GL path). A request from another thread (the
// RmlUi stair-size row) names a model-id range; we drop those uploads at BeginPass — before this
// frame records any SoH3D draws — so the next draw re-uploads from the (already refreshed) CPU
// model. A full vkDeviceWaitIdle makes the destroy safe; it only happens on a config change.
static int g_evictLo = 0, g_evictHi = 0;
static bool g_evictPending = false;
extern "C" void SoH3D_Vk_RequestEvictRange(int lo, int hi) {
    g_evictLo = lo; g_evictHi = hi; g_evictPending = true;
}
static void applyPendingEvict() {
    if (!g_evictPending || g_device == VK_NULL_HANDLE)
        return;
    g_evictPending = false;
    vkDeviceWaitIdle(g_device);
    for (auto it = g_models.begin(); it != g_models.end();) {
        if (it->first >= g_evictLo && it->first < g_evictHi) {
            VkModel& m = it->second;
            for (auto& t : m.textures) {
                if (t.view) vkDestroyImageView(g_device, t.view, nullptr);
                if (t.image) vkDestroyImage(g_device, t.image, nullptr);
                if (t.mem) vkFreeMemory(g_device, t.mem, nullptr);
            }
            if (m.vbo) vkDestroyBuffer(g_device, m.vbo, nullptr);
            if (m.vboMem) vkFreeMemory(g_device, m.vboMem, nullptr);
            it = g_models.erase(it);
        } else {
            ++it;
        }
    }
}

extern "C" void SoH3D_Vk_BeginPass(void) {
    g_ctxValid = false;
    if (!Fast::g_activeVulkanApi)
        return;
    if (!Fast::g_activeVulkanApi->BeginSoH3DPass(g_ctx))
        return;
    if (!ensureResources(g_ctx))
        return;
    applyPendingEvict(); // drop any models flagged for reload (e.g. stair size changed) before drawing
    g_ctxValid = true;
    // Reset this frame-in-flight's UBO ring + descriptor pool (the backend's in-flight fence,
    // waited at StartFrame, guarantees the previous use of this index has completed).
    Ring& r = g_rings[g_ctx.frameIndex];
    r.offset = 0;
    vkResetDescriptorPool(g_device, r.pool, 0);
}

extern "C" void SoH3D_Vk_DrawModel(int modelId, const float* mp16, const float* mv16, int lit, int invertY,
                                   unsigned char r8, unsigned char g8, unsigned char b8, unsigned char a8,
                                   float aspectAdj, const float* boneData, int boneCnt,
                                   unsigned long long midMask, int sky, float uvOffU, float uvOffV) {
    if (!g_ctxValid)
        return;
    VkModel* m = ensureUploaded(modelId);
    if (!m)
        return;

    Ring& ring = g_rings[g_ctx.frameIndex];
    VkCommandBuffer cmd = g_ctx.cmd;

    // Base UBO fields shared by all groups of this draw (per-group alphaRef/depthOffset patched below).
    VkUbo base{};
    memcpy(base.uMP, mp16, sizeof(base.uMP));
    base.uMP[0] *= aspectAdj; // mirror Fast3D's AdjXForAspectRatio (MP column 0, row-major 0/4/8/12)
    base.uMP[4] *= aspectAdj;
    base.uMP[8] *= aspectAdj;
    base.uMP[12] *= aspectAdj;
    memcpy(base.uMV, mv16, sizeof(base.uMV));
    for (int k = 0; k < 32; k++)
        for (int e = 0; e < 16; e++)
            base.uBones[k * 16 + e] = (e % 5 == 0) ? 1.0f : 0.0f; // identity
    if (boneData && boneCnt > 0) {
        // boneData is row-major (M*v). GL uploads it with glUniformMatrix4fv(..., GL_TRUE, ...) which
        // transposes on upload, so GLSL stores M itself. Vulkan std140 mat4 is column-major with no
        // transpose-on-upload, and a raw memcpy of row-major data stores M^T -> wrong skinning. So
        // transpose each bone matrix CPU-side to match GL exactly.
        int nb = boneCnt < 32 ? boneCnt : 32;
        for (int k = 0; k < nb; k++) {
            const float* s = boneData + k * 16;
            float* d = base.uBones + k * 16;
            for (int r = 0; r < 4; r++)
                for (int col = 0; col < 4; col++)
                    d[col * 4 + r] = s[r * 4 + col];
        }
    }
    base.uParams[0] = invertY ? -1.0f : 1.0f;
    base.uParams[1] = lit ? 1.0f : 0.0f;
    base.uTintSkin[0] = r8 / 255.0f;
    base.uTintSkin[1] = g8 / 255.0f;
    base.uTintSkin[2] = b8 / 255.0f;
    base.uTintSkin[3] = (boneData && boneCnt > 0) ? 1.0f : 0.0f;

    // World-space sun direction (set per frame by soh3d.c into the GL pass's global).
    base.uLightDir[0] = gSoH3dLightDirWorld[0];
    base.uLightDir[1] = gSoH3dLightDirWorld[1];
    base.uLightDir[2] = gSoH3dLightDirWorld[2];
    base.uLightDir[3] = sky ? 1.0f : 0.0f; // skybox dome: pin to far plane in the vertex shader
    base.uExtra[0] = a8 / 255.0f;          // per-draw opacity (dawn/dusk dome cross-fade); 1 = opaque
    base.uExtra[1] = uvOffU;               // texcoord scroll U (cloud-band drift, #28b); 0 = none
    base.uExtra[2] = uvOffV;               // texcoord scroll V
    bool forceBlend = (a8 < 255);          // translucent draw -> alpha-over even if the material is opaque

    bool vboBound = false;
    for (const VkGroup& grp : m->groups) {
        if (grp.cull)
            continue;
        if (grp.meshId >= 0 && grp.meshId < 64 && !((midMask >> grp.meshId) & 1ull))
            continue;
        if (ring.offset + g_uboStride > ring.capacity)
            return; // ring exhausted this frame

        VkUbo ubo = base;
        ubo.uParams[2] = grp.alphaTest ? grp.alphaRef : 0.0f;
        ubo.uParams[3] = grp.polygonOffset;
        const VkDeviceSize uboOff = ring.offset;
        memcpy((uint8_t*)ring.mapped + uboOff, &ubo, sizeof(ubo));
        ring.offset += g_uboStride;

        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = ring.pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &g_setLayout;
        VkDescriptorSet set;
        if (vkAllocateDescriptorSets(g_device, &dai, &set) != VK_SUCCESS)
            return;

        VkDescriptorBufferInfo bi{};
        bi.buffer = ring.ubo;
        bi.offset = uboOff;
        bi.range = sizeof(VkUbo);

        VkImageView view = g_dummyTex.view;
        VkSampler samp = g_dummySampler;
        if (grp.texIndex >= 0 && grp.texIndex < (int)m->textures.size()) {
            view = m->textures[grp.texIndex].view;
            samp = getSampler(grp.wrapS, grp.wrapT);
        }
        VkDescriptorImageInfo ii{};
        ii.sampler = samp;
        ii.imageView = view;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = set;
        w[0].dstBinding = 0;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[0].descriptorCount = 1;
        w[0].pBufferInfo = &bi;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = set;
        w[1].dstBinding = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[1].descriptorCount = 1;
        w[1].pImageInfo = &ii;
        vkUpdateDescriptorSets(g_device, 2, w, 0, nullptr);

        // Translucent draw over an opaque material: synthesize a standard alpha-over pipeline (the
        // VkGroup blend-factor defaults are SRC_ALPHA / ONE_MINUS_SRC_ALPHA) so uExtra.x composites,
        // mirroring the GL path's forceBlend. Depth-write/offset/etc. are inherited from the group.
        VkGroup gb = grp;
        if (forceBlend && !grp.blendEnable) {
            gb.blendEnable = 1;
            gb.bSrcRGB = 0x0302; gb.bDstRGB = 0x0303; gb.bEqRGB = 0x8006; // SRC_ALPHA / 1-SRC_ALPHA / ADD
            gb.bSrcA = 0x0302;   gb.bDstA = 0x0303;   gb.bEqA = 0x8006;
        }
        // Front-face winding flips with invertY (clip.y negated in the vertex shader); the flip
        // toggle lets the correct convention be found live. See the GL backend's drawOne.
        int frontCW = (invertY != 0) ^ (gSoH3dFaceCullFlip != 0);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipeline(gb, frontCW));
        vkCmdSetViewport(cmd, 0, 1, &g_ctx.viewport);
        vkCmdSetScissor(cmd, 0, 1, &g_ctx.scissor);
        vkCmdSetBlendConstants(cmd, grp.blendColor);
        if (!vboBound) {
            VkDeviceSize zero = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &m->vbo, &zero);
            vboBound = true;
        }
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeLayout, 0, 1, &set, 0, nullptr);
        vkCmdDraw(cmd, grp.count, 1, grp.first, 0);
    }
}

extern "C" void SoH3D_Vk_EndPass(void) {
    g_ctxValid = false;
}

#endif // ENABLE_VULKAN
