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

option(ENABLE_UNIT_TESTS "Enable the Unit tests" On)
set(ENABLE_FLOG ${FleCSI_ENABLE_FLOG})
set(FLECSI_RUNTIME_MODEL ${FleCSI_RUNTIME_MODEL})
set(FLECSI_BACKEND ${FleCSI_BACKEND})
include(FleCSI/unit)

if(ENABLE_UNIT_TESTS)
  flecsi_enable_testing()

  # Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
  endif()

  include(FetchContent)
  FetchContent_Declare(494_bus
	URL https://www.cise.ufl.edu/research/sparse/MM/HB/494_bus.tar.gz
	URL_HASH SHA256=9b2d33c6fe52ee5b4980da1a2d2b78201924a36714234757677fa5c829c483ca)
  FetchContent_Declare(chem97ztz
	URL https://www.cise.ufl.edu/research/sparse/MM/Bates/Chem97ZtZ.tar.gz
	URL_HASH SHA256=bb7de09a73368e07e48f19c288743b04b220aaf28cc2b7fcfef6f1d47440a2bb)
  FetchContent_Declare(psmigr_3
	URL https://www.cise.ufl.edu/research/sparse/MM/HB/psmigr_3.tar.gz
	URL_HASH SHA256=bf45ebfdb2837cdd40fa934637170d6989df1f42c81e9c9da0a88afee0b3bcd6)
  FetchContent_MakeAvailable(494_bus Chem97ZtZ psmigr_3)
endif()

include(FleCSI/summary)
option(ENABLE_FORMAT "Enable format target" OFF)
mark_as_advanced(ENABLE_FORMAT)

if(ENABLE_FORMAT)
  include(FleCSI/format)

  set(ClangFormat_VERSION "13.0.1" CACHE STRING
      "Set the required version (major[.minor[.patch]]) of clang-format")
  mark_as_advanced(ClangFormat_VERSION)

  flecsi_add_format_target(${PROJECT_NAME} ${PROJECT_SOURCE_DIR} "${ClangFormat_VERSION}")
endif()

# option to build examples
option(FLECSOLVE_BUILD_EXAMPLES "Build the examples" OFF)

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
if(ENABLE_FORMAT AND TARGET ClangFormat::ClangFormat)
  flecsi_summary_info("ClangFormat_EXECUTABLE" "${ClangFormat_EXECUTABLE}" TRUE)
endif()
string(APPEND _summary "\n")
flecsi_summary_info("FLECSI_BACKEND" "${FleCSI_BACKEND}" TRUE)
flecsi_summary_option("ENABLE_FLOG" ${ENABLE_FLOG} "")
flecsi_summary_option("ENABLE_UNIT_TESTS" ${ENABLE_UNIT_TESTS} "")
flecsi_summary_option("FLECSOLVE_BUILD_EXAMPLES" ${FLECSOLVE_BUILD_EXAMPLES} "")

message(STATUS ${_summary})
