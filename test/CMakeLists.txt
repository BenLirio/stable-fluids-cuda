cmake_minimum_required(VERSION 3.16)
project(MyCudaOpenMPProject LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

set(SOURCE_FILES main.cu)
add_executable(MyCudaOpenMPProject ${SOURCE_FILES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(MyCudaOpenMPProject PUBLIC OpenMP::OpenMP_CXX)
endif()

# Pass the OpenMP flags to the host compiler when using NVCC
set_property(TARGET MyCudaOpenMPProject PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_options(MyCudaOpenMPProject PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
