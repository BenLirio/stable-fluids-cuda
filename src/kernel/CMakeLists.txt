file(GLOB KERNEL_SOURCES "*.cu")

add_library(kernel ${KERNEL_SOURCES})