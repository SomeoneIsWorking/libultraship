// SoH3D Vulkan render pass — the Vulkan counterpart of soh3d_gl.cpp's GPU work.
//
// The backend-agnostic bookkeeping (the draw list, per-emit pose capture, the light/shadow/AO
// tunables, and the C-ABI entry points SoH3D_GL_*) stays in soh3d_gl.cpp; when the live Fast3D
// backend is the Vulkan one, those entry points dispatch the actual GPU submission here. This
// module owns its own Vulkan resources (per-model vertex buffers + textures, the model pipeline,
// per-draw uniform/bone ring) and records its draws into the backend's current command buffer and
// render pass (GfxRenderingAPIVulkan::BeginSoH3DPass), so the OoT3D content interleaves
// depth-correctly with the N64 geometry.
//
// NOTE: shadows + screen-space AO (the GL pass's extra offscreen passes) are not ported here yet;
// this renders the core textured / skinned / half-Lambert-lit content. Those enhancements are a
// follow-up. Build-structure NOTE: the C-ABI entry points currently live in soh3d_gl.cpp (compiled
// only with ENABLE_OPENGL); a no-GL build (macOS) will need that bookkeeping extracted to a shared
// TU. See the dispatch block in soh3d_gl.cpp.
#pragma once
#include "fast/soh3d_gl.h"

#ifdef __cplusplus
extern "C" {
#endif

// 1 if the live Fast3D backend is the Vulkan one (so the GL entry points should dispatch here).
int SoH3D_Vk_Active(void);

// Mirror of SoH3D_GL_SetModelProvider for the Vulkan model store (the GL setter forwards to this).
void SoH3D_Vk_SetProvider(SoH3DModelProvider fn);

// Bracket the SoH3D Vulkan pass: BeginPass opens the backend's current render pass; DrawModel
// records one (already pose-resolved) model; EndPass is a no-op placeholder for symmetry. The
// caller (SoH3D_GL_RenderPass) does the per-item pose interpolation, exactly as for the GL path.
void SoH3D_Vk_BeginPass(void);
void SoH3D_Vk_DrawModel(int modelId, const float* mp16, const float* mv16, int lit, int invertY,
                        unsigned char r, unsigned char g, unsigned char b, unsigned char a, float aspectAdj,
                        const float* boneData, int boneCnt, unsigned long long midMask, int sky,
                        float uvOffU, float uvOffV);
void SoH3D_Vk_EndPass(void);

// Mirror of SoH3D_GL_RequestEvictRange for the Vulkan model store (the GL request forwards here):
// drop cached uploads with id in [lo,hi) at the next BeginPass so they re-upload at the new size.
void SoH3D_Vk_RequestEvictRange(int lo, int hi);

#ifdef __cplusplus
}
#endif
