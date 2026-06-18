#include "SohRmlUi.h"

#include "RmlUi_Renderer_GL3.h"
#include "RmlUi_Platform_SDL.h"
#include "RmlRenderInterfaceVk.h"
// Pull in the bundled glad GL loader (declarations only; the implementation lives in
// RmlUi_Renderer_GL3.cpp's translation unit). Gives us glGet*/glBind* for the state guard.
#include "RmlUi_Include_GL3.h"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Input.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_events.h>

#include "ship/Context.h"
#include <spdlog/spdlog.h>

namespace Ship {

// RmlUi runtime is process-global (Rml::Initialise / Rml::Shutdown). Track init so a second
// SohRmlUi (e.g. after a backend switch) does not double-initialise the library.
static bool sRmlLibraryInitialised = false;

SohRmlUi::SohRmlUi() = default;

SohRmlUi::~SohRmlUi() {
    Shutdown();
}

bool SohRmlUi::Init(void* sdlWindow, void* glContext, int width, int height, bool vulkan) {
    if (mInitialised) {
        return true;
    }

    mSdlWindow = sdlWindow;
    mVulkan = vulkan;
    mWidth = width > 0 ? width : 1;
    mHeight = height > 0 ? height : 1;

    Rml::String gl_message = "Vulkan";
    Rml::RenderInterface* renderInterface = nullptr;
    if (mVulkan) {
#ifdef ENABLE_VULKAN
        mVkRenderInterface = std::make_unique<RmlRenderInterfaceVk>();
        renderInterface = mVkRenderInterface.get();
#else
        SPDLOG_ERROR("[SohRmlUi] Vulkan requested but ENABLE_VULKAN is off");
        return false;
#endif
    } else {
        // Load the bundled glad GL functions against the already-current Fast3D GL context.
        if (!RmlGL3::Initialize(&gl_message)) {
            SPDLOG_ERROR("[SohRmlUi] RmlGL3::Initialize failed: {}", gl_message);
            return false;
        }
        mRenderInterface = std::make_unique<RenderInterface_GL3>();
        if (!*mRenderInterface) {
            SPDLOG_ERROR("[SohRmlUi] Failed to construct the GL3 render interface");
            mRenderInterface.reset();
            return false;
        }
        renderInterface = mRenderInterface.get();
    }

    mSystemInterface = std::make_unique<SystemInterface_SDL>();
    mSystemInterface->SetWindow(static_cast<SDL_Window*>(sdlWindow));

    // Interfaces must be installed before Rml::Initialise().
    Rml::SetSystemInterface(mSystemInterface.get());
    Rml::SetRenderInterface(renderInterface);

    if (!sRmlLibraryInitialised) {
        if (!Rml::Initialise()) {
            SPDLOG_ERROR("[SohRmlUi] Rml::Initialise failed");
            mRenderInterface.reset();
            mSystemInterface.reset();
            return false;
        }
        sRmlLibraryInitialised = true;
    }

    // Font + document live next to the executable (copied there at build time). Absolute paths
    // resolve through RmlUi's default file interface regardless of the process working directory.
    const std::string fontPath = Context::GetPathRelativeToAppBundle("assets/rml/LatoLatin-Regular.ttf");
    if (!Rml::LoadFontFace(fontPath, true)) {
        SPDLOG_ERROR("[SohRmlUi] Failed to load font face: {}", fontPath);
    }

    if (!mVulkan) {
        mRenderInterface->SetViewport(mWidth, mHeight);
    }
    mContext = Rml::CreateContext("soh3d", Rml::Vector2i(mWidth, mHeight));
    if (!mContext) {
        SPDLOG_ERROR("[SohRmlUi] Rml::CreateContext failed");
        Shutdown();
        return false;
    }

    const std::string docPath = Context::GetPathRelativeToAppBundle("assets/rml/soh3d_test.rml");
    mDocument = mContext->LoadDocument(docPath);
    if (!mDocument) {
        SPDLOG_ERROR("[SohRmlUi] Failed to load document: {}", docPath);
        Shutdown();
        return false;
    }
    // Start hidden; the menu is shown on demand via ToggleVisible() (Phase 2). The document stays
    // loaded either way — we gate update/render + input on mVisible.
    mDocument->Show();

    SPDLOG_INFO("[SohRmlUi] RmlUi initialised ({}x{}) — {} ({})", mWidth, mHeight, docPath, gl_message);
    mInitialised = true;
    // Debug: open the menu at startup (deterministic verification via the screenshot harness, no
    // input injection needed). Normal use opens it with ESC / the Start button.
    if (const char* e = std::getenv("SOH3D_RMLUI_OPEN"); e && e[0] == '1') {
        SetVisible(true);
    }
    return true;
}

void SohRmlUi::SetVisible(bool visible) {
    if (visible == mVisible) {
        return;
    }
    mVisible = visible;
    if (mVisible && mContext) {
        // Update once so layout is current, then drop focus onto the first focusable element so a
        // controller/keyboard can drive it immediately (matches Dusklight opening with a default focus).
        // Focus the element directly rather than simulating Tab: at open time (e.g. the startup
        // auto-open) a synthesised Tab does not reliably land on the first item, leaving nothing
        // highlighted. QuerySelector finds the first element opted into focus via tabindex.
        mContext->Update();
        Rml::Element* first = nullptr;
        if (mDocument) {
            // Match the first element that opts INTO focus (tabindex="auto"); a bare [tabindex]
            // selector would also match items opted out with tabindex="none" (e.g. the tabs),
            // and focusing one of those leaves nothing visibly highlighted.
            first = mDocument->QuerySelector("[tabindex='auto']");
        }
        if (first) {
            first->Focus();
        } else {
            FocusNext();
        }
    } else if (mContext) {
        if (Rml::Element* focus = mContext->GetFocusElement()) {
            focus->Blur();
        }
    }
}

void SohRmlUi::FocusNext() {
    if (mContext) {
        mContext->ProcessKeyDown(Rml::Input::KI_TAB, 0);
        mContext->ProcessKeyUp(Rml::Input::KI_TAB, 0);
    }
}

void SohRmlUi::FocusPrev() {
    if (mContext) {
        mContext->ProcessKeyDown(Rml::Input::KI_TAB, Rml::Input::KM_SHIFT);
        mContext->ProcessKeyUp(Rml::Input::KI_TAB, Rml::Input::KM_SHIFT);
    }
}

void SohRmlUi::ActivateFocused() {
    if (!mContext) {
        return;
    }
    Rml::Element* focus = mContext->GetFocusElement();
    if (!focus) {
        return;
    }
    // If the focused element is a container row (the whole row takes focus for a clear highlight),
    // toggle the control it wraps; otherwise activate the focused element directly. This lets a
    // controller "A"/Enter flip a checkbox while focus rests on the readable row, not the tiny box.
    if (Rml::Element* control = focus->QuerySelector("input, select, button")) {
        control->Click();
    } else {
        focus->Click();
    }
}

bool SohRmlUi::ProcessSdlEvent(void* sdlEvent) {
    if (!mInitialised || !mContext || !sdlEvent) {
        return false;
    }
    SDL_Event& ev = *static_cast<SDL_Event*>(sdlEvent);

    // Toggle bindings are always live (so the menu can be opened/closed): ESC on the keyboard, or
    // the Start button on a game controller. (F1 is SoH's existing ImGui menu, left alone.)
    if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE && ev.key.repeat == 0) {
        ToggleVisible();
        return true;
    }
    if (ev.type == SDL_CONTROLLERBUTTONDOWN && ev.cbutton.button == SDL_CONTROLLER_BUTTON_START) {
        ToggleVisible();
        return true;
    }

    if (!mVisible) {
        return false;
    }

    // Menu is open: map directional input to focus nav, A/Enter to activate, B/Esc to close; pass
    // everything else (mouse, text, other keys) through the SDL platform shim. Consume it all so the
    // game does not also act on input while the menu is up.
    switch (ev.type) {
        case SDL_KEYDOWN:
            switch (ev.key.keysym.sym) {
                case SDLK_DOWN:
                case SDLK_RIGHT:
                    FocusNext();
                    return true;
                case SDLK_UP:
                case SDLK_LEFT:
                    FocusPrev();
                    return true;
                case SDLK_RETURN:
                case SDLK_KP_ENTER:
                case SDLK_SPACE:
                    ActivateFocused();
                    return true;
                default:
                    RmlSDL::InputEventHandler(mContext, static_cast<SDL_Window*>(mSdlWindow), ev);
                    return true;
            }
        case SDL_CONTROLLERBUTTONDOWN:
            switch (ev.cbutton.button) {
                case SDL_CONTROLLER_BUTTON_DPAD_DOWN:
                case SDL_CONTROLLER_BUTTON_DPAD_RIGHT:
                    FocusNext();
                    return true;
                case SDL_CONTROLLER_BUTTON_DPAD_UP:
                case SDL_CONTROLLER_BUTTON_DPAD_LEFT:
                    FocusPrev();
                    return true;
                case SDL_CONTROLLER_BUTTON_A:
                    ActivateFocused();
                    return true;
                case SDL_CONTROLLER_BUTTON_B:
                    SetVisible(false);
                    return true;
                default:
                    return true;
            }
        case SDL_MOUSEMOTION:
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
        case SDL_MOUSEWHEEL:
        case SDL_TEXTINPUT:
        case SDL_KEYUP:
            RmlSDL::InputEventHandler(mContext, static_cast<SDL_Window*>(mSdlWindow), ev);
            return true;
        default:
            return false;
    }
}

void SohRmlUi::Resize(int width, int height) {
    if (width <= 0 || height <= 0 || (width == mWidth && height == mHeight)) {
        return;
    }
    mWidth = width;
    mHeight = height;
    if (mContext) {
        mContext->SetDimensions(Rml::Vector2i(mWidth, mHeight));
    }
}

void SohRmlUi::UpdateAndRender() {
    if (!mInitialised || !mContext || !mVisible) {
        return;
    }

    // Track the live drawable size so the context + render target follow window resizes.
    if (mSdlWindow) {
        int dw = 0, dh = 0;
        SDL_GL_GetDrawableSize(static_cast<SDL_Window*>(mSdlWindow), &dw, &dh);
        Resize(dw, dh);
    }

    mContext->Update();

#ifdef ENABLE_VULKAN
    if (mVulkan) {
        // Record the menu into the Fast3D Vulkan backend's current pass (no GL state to guard).
        if (mVkRenderInterface && mVkRenderInterface->BeginFrame()) {
            mContext->Render();
            mVkRenderInterface->EndFrame();
        }
        return;
    }
#endif

    // --- GL state guard (see soh3d_gl.cpp for the same discipline) ---------------------------
    // RmlUi's BeginFrame/EndFrame already save+restore the render state it touches (blend,
    // scissor, stencil, depth, cull, viewport, colour mask, active texture). It does NOT restore
    // the bound VAO / program / array buffer / framebuffer / texture, which Fast3D caches across
    // draws — so snapshot and hand those back, then deterministically reset the blend func to
    // gfx_opengl's permanent init assumption so its implicit cache stays consistent with GL.
    GLint prevFbo = 0, prevVao = 0, prevProg = 0, prevArrBuf = 0, prevActiveTex = 0, prevTex0 = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProg);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArrBuf);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

    mRenderInterface->SetViewport(mWidth, mHeight);
    mRenderInterface->BeginFrame();
    mContext->Render();
    mRenderInterface->EndFrame();

    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prevFbo);
    glBindVertexArray((GLuint)prevVao);
    glUseProgram((GLuint)prevProg);
    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)prevArrBuf);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, (GLuint)prevTex0);
    glActiveTexture((GLenum)prevActiveTex);
    // gfx_opengl sets blend func/equation ONCE at init and relies on it persisting; RmlUi changed
    // it during its pass, so restore that permanent assumption deterministically.
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);
}

void SohRmlUi::Shutdown() {
    if (mContext) {
        Rml::RemoveContext(mContext->GetName());
        mContext = nullptr;
        mDocument = nullptr;
    }
    if (sRmlLibraryInitialised) {
        Rml::Shutdown(); // releases textures/geometry through the render interface; keep it alive here
        sRmlLibraryInitialised = false;
    }
    mRenderInterface.reset();
    mVkRenderInterface.reset();
    mSystemInterface.reset();
    mInitialised = false;
}

} // namespace Ship
