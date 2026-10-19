# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

cmake_minimum_required(VERSION 3.16)
project(dexcontrol_examples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The examples build against either a packaged SDK (tools/package_capi.sh,
# found through find_package) or the in-tree static library
# (`cargo build -p dexcontrol-capi`), whichever is available.
option(DEXCONTROL_IN_TREE "Link the in-tree libdexcontrol_capi.a" OFF)
set(DEXCONTROL_CARGO_PROFILE "" CACHE STRING
    "Cargo profile directory to link in-tree (default: debug)")

if(NOT DEXCONTROL_IN_TREE)
    find_package(dexcontrol QUIET)
endif()
if(dexcontrol_FOUND AND NOT DEXCONTROL_IN_TREE)
    set(DEXCONTROL_TARGET dexcontrol::dexcontrol)
    message(STATUS "dexcontrol examples: packaged SDK ${dexcontrol_VERSION}")
else()
    get_filename_component(_repo "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
    # Match the documented cargo build command; never silently use a stale
    # release archive just because it exists. Benchmarks opt into release.
    if(DEXCONTROL_CARGO_PROFILE)
        set(_profile "${DEXCONTROL_CARGO_PROFILE}")
    else()
        set(_profile "debug")
    endif()
    set(DEXCONTROL_CARGO_TARGET_DIR "${_repo}/target" CACHE PATH "Cargo target directory")
    set(DEXCONTROL_CAPI_LIBRARY "" CACHE FILEPATH "Exact Rust C API static library")
    if(DEXCONTROL_CAPI_LIBRARY)
        set(_lib "${DEXCONTROL_CAPI_LIBRARY}")
    else()
        set(_lib "${DEXCONTROL_CARGO_TARGET_DIR}/${_profile}/libdexcontrol_capi.a")
    endif()
    if(NOT EXISTS "${_lib}")
        message(FATAL_ERROR
            "dexcontrol not found: install the packaged SDK, or build the in-tree "
            "library with `cargo build -p dexcontrol-capi` (expected ${_lib})")
    endif()
    # The static Rust archive's platform dependencies.
    set(_platform_libs "dl;pthread;m")
    if(APPLE)
        set(_platform_libs "${_platform_libs};resolv;-framework Security;-framework CoreFoundation")
    endif()
    add_library(dexcontrol_in_tree STATIC IMPORTED)
    set_target_properties(dexcontrol_in_tree PROPERTIES
        IMPORTED_LOCATION "${_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${_repo}/bindings/c/include;${_repo}/bindings/cpp/include"
        INTERFACE_LINK_LIBRARIES "${_platform_libs}")
    set(DEXCONTROL_TARGET dexcontrol_in_tree)
    message(STATUS "dexcontrol examples: in-tree ${_profile} ${_lib}")
endif()

# A packaged control SDK remains usable with only a C++ toolchain. The
# optional viewer has its own native build and can be enabled explicitly.
set(_viewer_default ON)
if(dexcontrol_FOUND AND NOT DEXCONTROL_IN_TREE)
    set(_viewer_default OFF)
endif()
option(DEXCONTROL_WITH_VIEWER "Build reusable Rerun image visualization" ${_viewer_default})
if(DEXCONTROL_WITH_VIEWER)
    if(DEXCONTROL_CARGO_PROFILE)
        set(DEXCONTROL_VIEWER_PROFILE "${DEXCONTROL_CARGO_PROFILE}" CACHE STRING "Viewer Cargo profile")
    endif()
    if(DEXCONTROL_CARGO_TARGET_DIR)
        set(DEXCONTROL_VIEWER_TARGET_DIR "${DEXCONTROL_CARGO_TARGET_DIR}" CACHE PATH "Viewer Cargo target directory")
    endif()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../crates/dexcontrol-viewer"
                     "${CMAKE_BINARY_DIR}/viewer")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Werror)
endif()

# One executable per source file, named <group>_<name> and written to
# bin/<group>/<name> so the layout mirrors examples/vega_1/python.
file(GLOB_RECURSE _sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" CONFIGURE_DEPENDS
    "read_joint_positions.cpp" "basic_examples/*.cpp" "advanced_examples/*.cpp" "benchmark/*.cpp" "troubleshooting/*.cpp")
foreach(_source IN LISTS _sources)
    get_filename_component(_dir "${_source}" DIRECTORY)
    get_filename_component(_name "${_source}" NAME_WE)
    if(_dir)
        string(REPLACE "/" "_" _target "${_dir}/${_name}")
    else()
        set(_target "${_name}")
    endif()
    add_executable("${_target}" "${_source}")
    target_include_directories("${_target}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common")
    target_link_libraries("${_target}" PRIVATE ${DEXCONTROL_TARGET})
    if(DEXCONTROL_WITH_VIEWER AND _name MATCHES "^monitor_(chassis_cameras|head_camera|wrist_cameras)$")
        target_compile_definitions("${_target}" PRIVATE DEXCONTROL_WITH_VIEWER)
        target_link_libraries("${_target}" PRIVATE dexcontrol::viewer)
    endif()
    set_target_properties("${_target}" PROPERTIES
        OUTPUT_NAME "${_name}"
        # A generator expression prevents multi-config generators from appending
        # Debug/Release; the launcher's build trees already separate profiles.
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${_dir}/$<0:>")
endforeach()
