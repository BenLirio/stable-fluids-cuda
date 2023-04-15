file(GLOB GOLD_SOURCES "*.cu")

add_library(gold ${GOLD_SOURCES})
target_compile_options(gold PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
set_target_properties(gold PROPERTIES CUDA_SEPARABLE_COMPILATION ON)