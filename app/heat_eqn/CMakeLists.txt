
set(APP_NAME "heat_eqn")

#------------------------------------------------------------------------------#
# Add an executable.
#------------------------------------------------------------------------------#
add_executable(${APP_NAME}
  src/main.cc
)

target_include_directories(
  ${APP_NAME}
  PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
)

#------------------------------------------------------------------------------#
# Add the FleCSI libraries and dependencies.
#------------------------------------------------------------------------------#
target_link_libraries(${APP_NAME}
PRIVATE
  FleCSI::linalg
)

add_custom_command(
        TARGET ${APP_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/heateqn.cfg
                ${CMAKE_CURRENT_BINARY_DIR}/heateqn.cfg)