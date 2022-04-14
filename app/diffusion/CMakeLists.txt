
set(APP_NAME "diffusion")

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
