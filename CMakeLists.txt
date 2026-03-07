cmake_minimum_required(VERSION 3.24)
project(tinyllama_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# RTX 5050 = Blackwell = sm_120. Override with -DCMAKE_CUDA_ARCHITECTURES=... if needed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# MSVC: big-obj needed for heavy kernel templates; /fp:fast for FP16/FP32 math.
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/O2>
                       $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
                       $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>)
endif()

add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>                      # keep source lines for ncu
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>                      # print reg/smem usage per kernel
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>     # MSVC 14.50 is newer than CUDA 12.8 knows
)

file(GLOB KERNEL_SRCS CONFIGURE_DEPENDS src/kernels/*.cu)

add_executable(inference
    src/main.cpp
    src/weights.cpp
    src/model.cu
    ${KERNEL_SRCS}
)

target_include_directories(inference PRIVATE
    include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(inference PRIVATE CUDA::cudart)
set_target_properties(inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Handy side target: dump PTX for qkv_proj.cu for the hand-tune step.
add_custom_target(qkv_ptx
    COMMAND ${CMAKE_CUDA_COMPILER}
            -arch=sm_120 -O3 --use_fast_math -ptx
            -I${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src/kernels/qkv_proj.cu
            -o ${CMAKE_BINARY_DIR}/qkv_proj.ptx
    COMMENT "Generating qkv_proj.ptx"
    VERBATIM
)
