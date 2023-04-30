cmake_minimum_required(VERSION 3.16)
project(stable-fluids-cuda)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

find_package(OpenMP REQUIRED)

# find_package(glfw3 REQUIRED)
# find_package(GLEW REQUIRED)
# find_package(OpenGL REQUIRED)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  giflib
  URL https://sourceforge.net/projects/giflib/files/giflib-5.2.1.tar.gz
)
FetchContent_MakeAvailable(giflib)

set(COMPILE_PARAMETERS
  NUM_STEPS
  WIDTH
  HEIGHT
  DIFFUSION_RATE
  VISCOSITY
  GAUSS_SEIDEL_ITERATIONS
  TIME_STEP
  OUTPUT
  USE_GOLD
  KERNEL_FLAGS
)
foreach(PARAM IN LISTS COMPILE_PARAMETERS)
  if(NOT DEFINED ${PARAM})
    message(FATAL_ERROR "${PARAM} is not defined")
  endif()
  add_definitions(-D${PARAM}=${${PARAM}})
endforeach()

include_directories(include)
add_subdirectory(src)
add_subdirectory(tests)