
include(GoogleTest)
enable_testing()

file(GLOB TEST_FILES */*.cc)

foreach(FILE IN LISTS TEST_FILES)
  get_filename_component(NAME ${FILE} NAME_WE)
  add_executable(${NAME} ${FILE})
  target_link_libraries(${NAME} GTest::gtest_main gold kernel util)
  # set_target_properties(${NAME} PROPERTIES LINKER_LANGUAGE CUDA)
  gtest_discover_tests(${NAME})
endforeach()

# foreach(FILE IN LISTS UTIL_TEST_FILES)
#   get_filename_component(NAME ${FILE} NAME_WE)
#   add_executable(${NAME} ${FILE})
#   target_link_libraries(${NAME} GTest::gtest_main gold util)
#   gtest_discover_tests(${NAME})
# endforeach()

# if(OPENMP_FOUND)
#   target_link_libraries(${PROJECT_NAME}-gold-diffuse-tests OpenMP::OpenMP_CXX)
#   target_link_libraries(${PROJECT_NAME}-gold-advect-tests OpenMP::OpenMP_CXX)
#   target_link_libraries(${PROJECT_NAME}-gold-project-tests OpenMP::OpenMP_CXX)
#   target_link_libraries(${PROJECT_NAME}-vec2-tests OpenMP::OpenMP_CXX)
#   target_link_libraries(${PROJECT_NAME}-idx2-tests OpenMP::OpenMP_CXX)
# endif()