cmake_minimum_required(VERSION 3.13...3.20)

project(flecsolve VERSION 0.0.1 LANGUAGES CXX C)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Version 9.0 of gnu compilers required!")
  endif()
endif()

find_package(FleCSI REQUIRED)

set(ENABLE_UNIT_TESTS ON)
set(ENABLE_FLOG ${FleCSI_ENABLE_FLOG})
set(FLECSI_RUNTIME_MODEL ${FleCSI_RUNTIME_MODEL})
set(FLECSI_BACKEND ${FleCSI_BACKEND})
include(FleCSI/unit)

include(FleCSI/summary)
set(ClangFormat_VERSION "8" CACHE STRING "Set the required version
  (major[.minor[.patch]]) of clang-format")
mark_as_advanced(ClangFormat_VERSION)
include(FleCSI/format)

# option to build examples
option(FLECSOLVE_BUILD_EXAMPLES "Build the examples" OFF)
mark_as_advanced(FLECSOLVE_BUILD_EXAMPLES)

if(NOT FORMAT_ONLY)
  include(FleCSI/library)
  add_library(flecsolve)
  add_library_target(flecsolve flecsolve
	VERSION         ${${PROJECT_NAME}_VERSION}
	EXPORT_TARGET   flecsolveTargets
	NAMESPACE       flecsolve
	LINK_PUBLIC     FleCSI::FleCSI MPI::MPI_CXX)

  if(FLECSOLVE_BUILD_EXAMPLES)
    message(STATUS "Adding examples directory")
    add_subdirectory(examples)
  endif()

  configure_file(${PROJECT_SOURCE_DIR}/config/flecsolveConfig.cmake.in
	${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/flecsolveConfig.cmake @ONLY)

  install(
	FILES ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/flecsolveConfig.cmake
	DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIBDIR}/cmake/flecsolve)

  flecsi_summary_header()
  flecsi_summary_info("CMAKE_BUILD_TYPE" "${CMAKE_BUILD_TYPE}" TRUE)
  flecsi_summary_info("CMAKE_INSTALL_PREFIX" "${CMAKE_INSTALL_PREFIX}" TRUE)
  string(APPEND _summary "\n")
  flecsi_summary_info("CMAKE_CXX_COMPILER" "${CMAKE_CXX_COMPILER}" TRUE)
  flecsi_summary_info("CMAKE_CXX_COMPILER_VERSION"
    "${CMAKE_CXX_COMPILER_VERSION}" TRUE)
  flecsi_summary_info("CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}" TRUE)
  flecsi_summary_info("CMAKE_C_COMPILER" "${CMAKE_C_COMPILER}" TRUE)
  flecsi_summary_info("CMAKE_C_COMPILER_VERSION" "${CMAKE_C_COMPILER_VERSION}" TRUE)
  flecsi_summary_info("CMAKE_C_FLAGS" "${CMAKE_C_FLAGS}" TRUE)
  if(ClangFormat_FOUND)
    flecsi_summary_info("ClangFormat_EXECUTABLE" "${ClangFormat_EXECUTABLE}" TRUE)
  endif()
  string(APPEND _summary "\n")
  flecsi_summary_info("FLECSI_BACKEND" "${FleCSI_BACKEND}" TRUE)
  flecsi_summary_option("ENABLE_FLOG" ${ENABLE_FLOG} "")
  flecsi_summary_option("ENABLE_UNIT_TESTS" ${ENABLE_UNIT_TESTS} "")
  flecsi_summary_option("FLECSOLVE_BUILD_EXAMPLES" ${FLECSOLVE_BUILD_EXAMPLES} "")
  # string(APPEND _summary "\n")

  message(STATUS ${_summary})
else()
  flecsi_summary_header()
  flecsi_summary_option("FORMAT_ONLY" ${FORMAT_ONLY} "")
  message(STATUS ${_summary})
endif()
