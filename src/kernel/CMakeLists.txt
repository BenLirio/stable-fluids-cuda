file(GLOB KERNEL_SOURCES "*.cu")

add_library(kernel ${KERNEL_SOURCES})
target_compile_options(kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
