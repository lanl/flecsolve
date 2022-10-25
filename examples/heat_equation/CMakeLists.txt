add_executable(heat-explicit explicit.cc heat.cc)
target_link_libraries(heat-explicit PRIVATE flecsolve)
configure_file(explicit.cfg ${CMAKE_CURRENT_BINARY_DIR}/explicit.cfg COPYONLY)

add_executable(heat-implicit implicit.cc heat.cc)
target_link_libraries(heat-implicit PRIVATE flecsolve)
configure_file(implicit.cfg ${CMAKE_CURRENT_BINARY_DIR}/implicit.cfg COPYONLY)
