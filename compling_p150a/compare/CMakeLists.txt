cmake_minimum_required(VERSION 3.20)
project(compare_eltwise_sfpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()

set(TT_METAL_HOME "$ENV{TT_METAL_HOME}")
set(TT_METAL_BUILD "${TT_METAL_HOME}/.build/gcc")
set(TT_METAL_API_DIR "${TT_METAL_HOME}/tt_metal/api")
set(TT_METAL_HOSTDEVCOMMON_API_DIR "${TT_METAL_HOME}/tt_metal/hostdevcommon/api")
set(TT_STL_INCLUDE_DIR "${TT_METAL_HOME}/tt_stl")
set(FMT_INCLUDE_DIR "${TT_METAL_HOME}/.cpmcache/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include")
set(NLOHMANN_INCLUDE_DIR "${TT_METAL_HOME}/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include")
set(TT_METAL_LIB "${TT_METAL_BUILD}/tt_metal/Release/libtt_metal.so")

message(STATUS "TT_METAL_HOME=${TT_METAL_HOME}")
message(STATUS "TT_METAL_BUILD=${TT_METAL_BUILD}")
message(STATUS "TT_METAL_API_DIR=${TT_METAL_API_DIR}")
message(STATUS "TT_METAL_HOSTDEVCOMMON_API_DIR=${TT_METAL_HOSTDEVCOMMON_API_DIR}")
message(STATUS "TT_STL_INCLUDE_DIR=${TT_STL_INCLUDE_DIR}")
message(STATUS "FMT_INCLUDE_DIR=${FMT_INCLUDE_DIR}")
message(STATUS "NLOHMANN_INCLUDE_DIR=${NLOHMANN_INCLUDE_DIR}")
message(STATUS "TT_METAL_LIB=${TT_METAL_LIB}")

if(NOT EXISTS "${TT_METAL_API_DIR}/tt-metalium/host_api.hpp")
    message(FATAL_ERROR "Could not find tt-metalium headers under ${TT_METAL_API_DIR}")
endif()

if(NOT EXISTS "${TT_METAL_HOSTDEVCOMMON_API_DIR}/hostdevcommon/common_values.hpp")
    message(FATAL_ERROR "Could not find hostdevcommon headers under ${TT_METAL_HOSTDEVCOMMON_API_DIR}")
endif()

if(NOT EXISTS "${TT_STL_INCLUDE_DIR}/tt_stl/strong_type.hpp")
    message(FATAL_ERROR "Could not find tt_stl headers under ${TT_STL_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${FMT_INCLUDE_DIR}/fmt/base.h")
    message(FATAL_ERROR "Could not find fmt/base.h under ${FMT_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${NLOHMANN_INCLUDE_DIR}/nlohmann/json_fwd.hpp")
    message(FATAL_ERROR "Could not find nlohmann/json_fwd.hpp under ${NLOHMANN_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${TT_METAL_LIB}")
    message(FATAL_ERROR "Could not find libtt_metal.so at ${TT_METAL_LIB}")
endif()

add_executable(compare_eltwise_sfpu compare_eltwise_sfpu.cpp)

target_include_directories(compare_eltwise_sfpu PRIVATE
    "${TT_METAL_HOME}"
    "${TT_METAL_HOME}/tt_metal"
    "${TT_METAL_API_DIR}"
    "${TT_METAL_API_DIR}/tt-metalium"
    "${TT_METAL_HOSTDEVCOMMON_API_DIR}"
    "${TT_STL_INCLUDE_DIR}"
    "${FMT_INCLUDE_DIR}"
    "${NLOHMANN_INCLUDE_DIR}"
)

target_link_libraries(compare_eltwise_sfpu PRIVATE
    "${TT_METAL_LIB}"
)

target_compile_options(compare_eltwise_sfpu PRIVATE
    -Wall -Wextra
)

set_target_properties(compare_eltwise_sfpu PROPERTIES
    BUILD_RPATH "${TT_METAL_BUILD}/tt_metal/Release;${TT_METAL_BUILD}/lib;${TT_METAL_BUILD}/tt_metal/third_party/umd/device/Release"
    INSTALL_RPATH "${TT_METAL_BUILD}/tt_metal/Release;${TT_METAL_BUILD}/lib;${TT_METAL_BUILD}/tt_metal/third_party/umd/device/Release"
)
