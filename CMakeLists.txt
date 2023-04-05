cmake_minimum_required(VERSION 3.16)
project(stable-fluids-cuda)
enable_language(CUDA)

add_executable(${PROJECT_NAME} main.cu gold.cu vec2.cu state.cu)