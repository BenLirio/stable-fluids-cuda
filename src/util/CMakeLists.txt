file(GLOB UTIL_SOURCES "*.cu")

add_library(util STATIC ${UTIL_SOURCES})
