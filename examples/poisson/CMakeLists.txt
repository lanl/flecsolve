add_executable(poisson poisson.cc)
target_link_libraries(poisson PRIVATE flecsolve)
configure_file(poisson.cfg ${CMAKE_CURRENT_BINARY_DIR}/poisson.cfg COPYONLY)
