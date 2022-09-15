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
find_package(FleCSICMake REQUIRED)

set(ENABLE_UNIT_TESTS ON)
set(ENABLE_FLOG ${FleCSI_ENABLE_FLOG})
set(FLECSI_RUNTIME_MODEL ${FleCSI_RUNTIME_MODEL})
set(FLECSI_BACKEND ${FleCSI_BACKEND})
include(unit)

include(summary)
set(ClangFormat_VERSION "8" CACHE STRING "Set the required version
  (major[.minor[.patch]]) of clang-format")
mark_as_advanced(ClangFormat_VERSION)
include(format)

# option to build examples
option(FLECSOLVE_BUILD_EXAMPLES "Build the examples" OFF)
mark_as_advanced(FLECSOLVE_BUILD_EXAMPLES)

if(NOT FORMAT_ONLY)
  include(library)
  add_library_target(flecsolve flecsolve
	VERSION         ${${PROJECT_NAME}_VERSION}
	EXPORT_TARGET   flecsolveTargets
	NAMESPACE       flecsolve
	LINK_PUBLIC     FleCSI::FleCSI MPI::MPI_CXX)

  if(FLECSOLVE_BUILD_EXAMPLES)
    message(STATUS "Adding examples directory")
    add_subdirectory(examples)
  endif()

  include(CMakePackageConfigHelpers)
  configure_package_config_file(cmake/flecsolveConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/flecsolveConfig.cmake
	INSTALL_DESTINATION lib/cmake/flecsolve
	)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/flecsolveConfigVersion.cmake
	COMPATIBILITY SameMajorVersion
	)

  install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/flecsolveConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/flecsolveConfigVersion.cmake
	DESTINATION lib/cmake/flecsolve
	)

  summary_header()
  summary_info("CMAKE_BUILD_TYPE" "${CMAKE_BUILD_TYPE}" TRUE)
  summary_info("CMAKE_INSTALL_PREFIX" "${CMAKE_INSTALL_PREFIX}" TRUE)
  string(APPEND _summary "\n")
  summary_info("CMAKE_CXX_COMPILER" "${CMAKE_CXX_COMPILER}" TRUE)
  summary_info("CMAKE_CXX_COMPILER_VERSION"
    "${CMAKE_CXX_COMPILER_VERSION}" TRUE)
  summary_info("CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}" TRUE)
  summary_info("CMAKE_C_COMPILER" "${CMAKE_C_COMPILER}" TRUE)
  summary_info("CMAKE_C_COMPILER_VERSION" "${CMAKE_C_COMPILER_VERSION}" TRUE)
  summary_info("CMAKE_C_FLAGS" "${CMAKE_C_FLAGS}" TRUE)
  if(ClangFormat_FOUND)
    summary_info("ClangFormat_EXECUTABLE" "${ClangFormat_EXECUTABLE}" TRUE)
  endif()
  string(APPEND _summary "\n")
  summary_info("FLECSI_BACKEND" "${FleCSI_BACKEND}" TRUE)
  summary_option("ENABLE_FLOG" ${ENABLE_FLOG} "")
  summary_option("ENABLE_UNIT_TESTS" ${ENABLE_UNIT_TESTS} "")
  # string(APPEND _summary "\n")

  message(STATUS ${_summary})
else()
  summary_header()
  summary_option("FORMAT_ONLY" ${FORMAT_ONLY} "")
  message(STATUS ${_summary})
endif()
