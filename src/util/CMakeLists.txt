file(GLOB UTIL_SOURCES "*.cu")

add_library(util STATIC ${UTIL_SOURCES})
target_link_libraries(util PRIVATE gif)
target_compile_options(util PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
set_target_properties(util PROPERTIES CUDA_SEPARABLE_COMPILATION ON)