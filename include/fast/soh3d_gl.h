// SoH3D direct-GL renderer (PC-native path for OoT3D models). Bypasses the
// Fast3D/N64 dlist+TMEM path entirely: models are uploaded to GL VBOs/textures
// once, then drawn with our own shader using the game's current MVP, INSIDE the
// scene pass (invoked from the OTR_G_SOH3D_DRAW dlist opcode, so GL is current and
// the scene depth buffer is intact -> correct occlusion).
//
// C linkage so the (C) soh game code and the C++ asset bridge can both call it.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One interleaved render vertex (model space). Matches SoH3D::CmbVertex layout.
// boneIds/weights drive GPU skinning: pos_skinned = sum_i weights[i] *
// uBones[boneIds[i]] * pos. With uBones = identity this is the bind pose (weights
// sum to 1), so a model with no animation set renders unchanged. Up to 4 bones/vtx.
typedef struct SoH3DGlVtx {
    float pos[3];
    float nrm[3];
    float uv[2];
    float boneIds[4];
    float weights[4];
    float color[4]; // per-vertex RGBA (OoT3D baked lighting / additive falloff)
} SoH3DGlVtx;

// Max bones in the skinning uniform array (covers OoT3D characters; childlink=25).
#define SOH3D_GL_MAX_BONES 32

// One per-material draw batch (triangle list).
typedef struct SoH3DGlGroup {
    const SoH3DGlVtx* verts;
    int vertCount;       // multiple of 3
    int texIndex;        // index into the model's textures, -1 = untextured
    int alphaTest;       // 0/1
    float alphaRef;      // [0,1] discard threshold when alphaTest
    unsigned wrapS, wrapT; // GL wrap enums (0x2901 REPEAT, 0x2900 CLAMP, ...)
    // Per-material blend state (GL enum values from the CMB, used verbatim). When
    // blendEnable is 0 the material is opaque (depth-write, no blend). Additive
    // light-shaft materials have blendDstRGB = GL_ONE; honoring this is what stops
    // them rendering as opaque trapezoids.
    int blendEnable;          // 0/1
    unsigned blendSrcRGB, blendDstRGB, blendEqRGB; // glBlendFuncSeparate / glBlendEquationSeparate
    unsigned blendSrcA, blendDstA, blendEqA;
    float blendColor[4];      // glBlendColor (for CONSTANT_COLOR/ALPHA factors)
    int depthWrite;           // 0/1 (translucent volumes disable depth write)
} SoH3DGlGroup;

// One decoded texture (RGBA8, w*h*4 bytes, row 0 = top).
typedef struct SoH3DGlTex {
    const unsigned char* rgba;
    int w, h;
} SoH3DGlTex;

// The game (soh) registers this to supply a model's CPU data ON DEMAND, the first
// time a model id is drawn (called with GL current, on the render thread). It must
// fill *groups/*groupCount and *texs/*texCount with arrays that stay valid for the
// duration of the call (the renderer uploads to GL immediately). Return 1 on
// success, 0 if the model id is unknown / failed to load.
typedef int (*SoH3DModelProvider)(int modelId, const SoH3DGlGroup** groups, int* groupCount,
                                  const SoH3DGlTex** texs, int* texCount);
void SoH3D_GL_SetModelProvider(SoH3DModelProvider fn);

// Draw a model by stable id (uploads lazily via the provider on first use).
// mp16 = the interpreter's current MP_matrix (row-major float[4][4]). invertY
// mirrors the target FBO's invertY (negate clip.y). tint multiplies the texture.
// aspectAdj = the per-vertex clip-space X scale Fast3D applies to N64 vertices
// (Interpreter::AdjXForAspectRatio: (4/3)/(w/h) for the resizable game FB, 1.0 for
// fixed-aspect FBs). The N64 actors get it; without applying the SAME factor here
// the OoT3D scene/models shear horizontally vs the N64 actors as the camera pans.
void SoH3D_GL_Draw(int modelId, const float* mp16, int invertY, unsigned char r, unsigned char g, unsigned char b,
                   float aspectAdj);

// Set the per-bone skinning matrices for a model (row-major float[16] each, indexed
// by bone id). Applied as the shader's uBones at the next draw. n is clamped to
// SOH3D_GL_MAX_BONES. Passing n==0 resets to the bind pose (identity). Cheap — just
// stores the matrices; call once per game frame after computing the animated pose.
void SoH3D_GL_SetBones(int modelId, const float* mats16, int n);

#ifdef __cplusplus
}
#endif
