#pragma once

#ifdef __cplusplus

#include <memory>

// Forward declarations keep the heavy RmlUi / SDL / GL headers out of this header so it
// can be included from Fast3dGui (and elsewhere) without dragging in the bundled glad
// loader or RmlUi's public surface.
class RenderInterface_GL3;
class SystemInterface_SDL;
namespace Rml {
class Context;
class ElementDocument;
} // namespace Rml

namespace Ship {

/**
 * @brief Owns the RmlUi runtime for the Fast3D OpenGL backend.
 *
 * Phase 1 integration spike: stands up the GL3 RenderInterface, the SDL SystemInterface and a
 * window-sized Rml::Context, loads a single styled test document and renders it over the game
 * each frame. Rendering is wrapped in GL state save/restore so it does not leak into the
 * Fast3D / ImGui passes (see the soh3d_gl pass for the same discipline).
 *
 * Input feeding and controller navigation are deliberately out of scope here (Phase 2).
 */
class SohRmlUi {
  public:
    // Constructor and destructor are defined out-of-line so the unique_ptr<> members can hold
    // incomplete types in this header (the RmlUi interfaces are only complete in the .cpp).
    SohRmlUi();
    ~SohRmlUi();

    SohRmlUi(const SohRmlUi&) = delete;
    SohRmlUi& operator=(const SohRmlUi&) = delete;

    /**
     * @brief Initialises RmlUi against the already-current SDL/GL context.
     * @param sdlWindow  SDL_Window* (as void*) owning the GL context.
     * @param glContext  SDL_GLContext (as void*), assumed current on the calling thread.
     * @param width      Initial window width in pixels.
     * @param height     Initial window height in pixels.
     * @return true if RmlUi initialised and the test document loaded.
     */
    bool Init(void* sdlWindow, void* glContext, int width, int height);

    /** @brief Updates and renders the RmlUi context, wrapped in GL state save/restore. */
    void UpdateAndRender();

    /** @brief Updates the context + render-interface viewport after a window resize. */
    void Resize(int width, int height);

    /** @brief Tears down the document, context and RmlUi runtime. */
    void Shutdown();

    bool IsInitialised() const {
        return mInitialised;
    }

  private:
    bool mInitialised = false;
    int mWidth = 0;
    int mHeight = 0;
    void* mSdlWindow = nullptr;

    std::unique_ptr<RenderInterface_GL3> mRenderInterface;
    std::unique_ptr<SystemInterface_SDL> mSystemInterface;
    Rml::Context* mContext = nullptr;          // owned by RmlUi, freed by Rml::Shutdown()
    Rml::ElementDocument* mDocument = nullptr;  // owned by the context
};

} // namespace Ship

#endif
