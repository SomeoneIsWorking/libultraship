# SoH3D dlist render harness — drives the Fast3D interpreter headlessly over a
# generated CMB->F3DEX2 model dlist (see dlist_harness.cpp).
#
# The model under test is the generated (ROM-derived, gitignored) crate source in
# soh/src/soh3d. It is compiled directly into the harness (raw Vtx/Gfx/tex arrays,
# no ResourceManager).

set(SOH3D_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../soh/src/soh3d")
set(SOH3D_MODEL_SRC "${SOH3D_DIR}/soh3d_kibako_model.c")

if(NOT EXISTS "${SOH3D_MODEL_SRC}")
    message(WARNING "soh3d_dlist_harness: ${SOH3D_MODEL_SRC} not found — "
                    "regenerate with tools/cmb_to_c.py. Skipping harness target.")
    return()
endif()

add_executable(soh3d_dlist_harness
    dlist_harness.cpp
    "${SOH3D_MODEL_SRC}"
)

target_include_directories(soh3d_dlist_harness PRIVATE "${SOH3D_DIR}")
set_property(TARGET soh3d_dlist_harness PROPERTY CXX_STANDARD 20)
# Must match libultraship's GBI ucode (set PRIVATE there, so it doesn't propagate):
# the gbi.h opcode macros are selected at compile time and must encode F3DEX2,
# or the interpreter walks the dlist with the wrong opcodes/stride.
target_compile_definitions(soh3d_dlist_harness PRIVATE F3DEX_GBI_2)
target_link_libraries(soh3d_dlist_harness PRIVATE libultraship)
