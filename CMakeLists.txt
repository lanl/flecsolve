cmake_minimum_required(VERSION 3.13...3.20)

project(flecsi-linalg VERSION 0.0.1 LANGUAGES CXX C)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(FLECSI_BACKEND mpi)
# set(ENABLE_FLOG ON)
# set(ENABLE_UNIT_TESTS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Version 9.0 of gnu compilers required!")
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "10.0")
    message(FATAL_ERROR "Cannot use this GCC version due to a regression:"
      "<https://gcc.gnu.org/bugzilla/show_bug.cgi?id=83258>")
  endif()
endif()

find_package(FleCSI REQUIRED)

find_package(FleCSICMake REQUIRED)
#list(APPEND CMAKE_MODULE_PATH ${FleCSICMake_PREFIX})

set(ENABLE_FLOG ${FleCSI_ENABLE_FLOG})
set(FLECSI_RUNTIME_MODEL ${FleCSI_RUNTIME_MODEL})
set(FLECSI_BACKEND ${FleCSI_BACKEND})
include(flog)
include(unit)

include(summary)
set(ClangFormat_VERSION "8" CACHE STRING "Set the required version
  (major[.minor[.patch]]) of clang-format")
mark_as_advanced(ClangFormat_VERSION)
include(format)

if(NOT FORMAT_ONLY)
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
