cmake_minimum_required(VERSION 3.13...3.20)

project(flecsi-linalg VERSION 0.0.1 LANGUAGES CXX C)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

find_package(FleCSI REQUIRED)

find_package(FleCSICMake REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${FleCSICMake_PREFIX})

set(ENABLE_FLOG ${FleCSI_ENABLE_FLOG})
set(FLECSI_RUNTIME_MODEL ${FleCSI_RUNTIME_MODEL})
include(flog)

set(FLECSI_UNIT_MAIN ${FleCSICMake_PREFIX}/unit-main.cc)
include(unit)

add_library(linalg INTERFACE)
add_library(FleCSI::linalg ALIAS linalg)

target_link_libraries(linalg INTERFACE FleCSI::FleCSI)

target_include_directories(linalg INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

add_subdirectory(flecsi-linalg)

install(TARGETS linalg EXPORT flecsi-linalgTargets
  INCLUDES DESITNATION include)

install(EXPORT flecsi-linalgTargets
  FILE flecsi-linalgTargets.cmake
  NAMESPACE FleCSI::
  DESTINATION lib/cmake/flecsi-linalg)

export(TARGETS linalg
  NAMESPACE FleCSI::
  FILE flecsi-linalgTargets.cmake)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flecsi-linalg
  DESTINATION include
  FILES_MATCHING PATTERN "*.hh")

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/flecsi-linalgConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/flecsi-linalgConfig.cmake
  INSTALL_DESTINATION lib/cmake/flecsi-linalg
)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/flecsi-linalgConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/flecsi-linalgConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/flecsi-linalgConfigVersion.cmake
  DESTINATION lib/cmake/flecsi-linalg
)
