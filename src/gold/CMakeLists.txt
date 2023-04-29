file(GLOB GOLD_SOURCES "*.cu")

add_library(gold ${GOLD_SOURCES})
set_target_properties(gold PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(OpenMP_CXX_FOUND)
  target_link_libraries(gold PUBLIC OpenMP::OpenMP_CXX)
endif()
if(OpenMP_C_FOUND)
  target_link_libraries(gold PUBLIC OpenMP::OpenMP_C)
endif()

set_property(TARGET gold PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_options(gold PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
target_compile_options(gold PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)