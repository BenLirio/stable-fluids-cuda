cmake_minimum_required(VERSION 3.16)
project(stable-fluids-cuda)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# NUM_STEPS
if(NOT DEFINED NUM_STEPS)
  message(FATAL_ERROR "NUM_STEPS is not defined")
endif()
add_definitions(-DNUM_STEPS=${NUM_STEPS})

# WIDTH
if(NOT DEFINED WIDTH)
  message(FATAL_ERROR "WIDTH is not defined")
endif()
add_definitions(-DWIDTH=${WIDTH})

# HEIGHT
if(NOT DEFINED HEIGHT)
  message(FATAL_ERROR "HEIGHT is not defined")
endif()
add_definitions(-DHEIGHT=${HEIGHT})

# DIFFUSION_RATE
if(NOT DEFINED DIFFUSION_RATE)
  message(FATAL_ERROR "DIFFUSION_RATE is not defined")
endif()
add_definitions(-DDIFFUSION_RATE=${DIFFUSION_RATE})

# VISCOSITY
if(NOT DEFINED VISCOSITY)
  message(FATAL_ERROR "VISCOSITY is not defined")
endif()
add_definitions(-DVISCOSITY=${VISCOSITY})

# GAUSS_SEIDEL_ITERATIONS
if(NOT DEFINED GAUSS_SEIDEL_ITERATIONS)
  message(FATAL_ERROR "GAUSS_SEIDEL_ITERATIONS is not defined")
endif()
add_definitions(-DGAUSS_SIDEL_ITERATIONS=${GAUSS_SEIDEL_ITERATIONS})

# TIME_STEP
if(NOT DEFINED TIME_STEP)
  message(FATAL_ERROR "TIME_STEP is not defined")
endif()
add_definitions(-DTIME_STEP=${TIME_STEP})


add_executable(${PROJECT_NAME} main.cu gold.cu vec2.cu state.cu idx2.cu)

enable_testing()
add_executable(${PROJECT_NAME}-tests tests.cc gold.cu vec2.cu state.cu idx2.cu)
target_link_libraries(${PROJECT_NAME}-tests GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}-tests)