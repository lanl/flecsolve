cmake_minimum_required(VERSION 3.23)

project(flecsolve VERSION 0.0.1 LANGUAGES CXX C)

set(PROJ flecsolve)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
set(flecsolve_INSTALL_DIR /tmp)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Version 9.0 of gnu compilers required!")
  endif()
endif()

find_package(FleCSI REQUIRED)
option(FLECSOLVE_ENABLE_AMP "Enable AMP wrapper in flecsolve" ON)
if(FLECSOLVE_ENABLE_AMP)
  find_package(AMP REQUIRED)
  find_package(TPLs REQUIRED)

  # amp data
  if( AMP_DATA OR AMP_DATA_URL )
	if( AMP_DATA_URL )
      message( STATUS "Downloading AMP Data - ${AMP_DATA_URL}" )
      get_filename_component( AMP_DATA "${AMP_DATA_URL}" NAME)
      setT( AMP_DATA "${CMAKE_CURRENT_BINARY_DIR}/${AMP_DATA}" )
      file( DOWNLOAD "${AMP_DATA_URL}" "${AMP_DATA}" )
	endif()
	if( "${AMP_DATA}" STREQUAL "" )
      unset( AMP_DATA )
	endif()
	if ( IS_DIRECTORY "${AMP_DATA}" )
      # AMP_DATA is a directory
	elseif( EXISTS "${AMP_DATA}" )
      # AMP_DATA is a file, try to unpack it
      execute_process(
		COMMAND ${CMAKE_COMMAND} -E tar xzf "${AMP_DATA}"
		WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      )
      if( EXISTS "${CMAKE_BINARY_DIR}/AMP-Data" )
		set( AMP_DATA "${CMAKE_BINARY_DIR}/AMP-Data" )
      else()
		message(FATAL_ERROR "Error unpacking tar file ${AMP_DATA}")
      endif()
	endif()
  endif()
  if( AMP_DATA )
	add_definitions( -DUSE_AMP_DATA )
  elseif( NOT TPLs_LIBMESH_FOUND AND NOT AMP_DATA )
	message(WARNING "LIBMESH not found or AMP_DATA is not set, some tests will be disabled" )
  endif()
endif()


option(FLECSOLVE_ENABLE_UNIT_TESTS "Enable the Unit tests" On)
include(FleCSI/unit)

if(FLECSOLVE_ENABLE_UNIT_TESTS)
  include(CTest)
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
  FetchContent_Declare(nos7
	URL https://www.cise.ufl.edu/research/sparse/MM/HB/nos7.tar.gz
	URL_HASH SHA256=c5d8d99bf4b54ee45e2f45d78530e3787f2e9670c000a68ad986a3b923e9e5ae)
  FetchContent_MakeAvailable(494_bus Chem97ZtZ psmigr_3 nos7)
endif()

include(FleCSI/summary)
option(FLECSOLVE_ENABLE_FORMAT "Enable format target" OFF)
mark_as_advanced(FLECSOLVE_ENABLE_FORMAT)

if(FLECSOLVE_ENABLE_FORMAT)
  include(FleCSI/format)

  set(ClangFormat_VERSION "13.0.1" CACHE STRING
      "Set the required version (major[.minor[.patch]]) of clang-format")
  mark_as_advanced(ClangFormat_VERSION)

  flecsi_add_format_target(${PROJECT_NAME} ${PROJECT_SOURCE_DIR} "${ClangFormat_VERSION}")
endif()

# option to build examples
option(FLECSOLVE_BUILD_EXAMPLES "Build the examples" OFF)

option(FLECSOLVE_BUILD_SHARED_LIBS "Build flecsolve with shared libraries" ON)
mark_as_advanced(FLECSOLVE_BUILD_SHARED_LIBS)
set(BUILD_SHARED_LIBS "${FLECSOLVE_BUILD_SHARED_LIBS}")

include(GNUInstallDirs)

add_library(flecsolve)
add_library(fecsolve::flecsolve ALIAS flecsolve)

target_include_directories(flecsolve
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${AMP_INCLUDE_DIRS} ${TPL_INCLUDE_DIRS}
)

target_link_libraries(flecsolve PUBLIC
  FleCSI::FleCSI MPI::MPI_CXX ${AMP_LIBRARIES} ${TPL_LIBRARIES})

add_subdirectory(flecsolve)

if(FLECSOLVE_BUILD_EXAMPLES)
  message(STATUS "Adding examples directory")
  add_subdirectory(examples)
endif()

configure_file(${PROJECT_SOURCE_DIR}/config/flecsolveConfig.cmake.in
  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/flecsolveConfig.cmake @ONLY)

# install
install(
  FILES ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/flecsolveConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flecsolve)

install(TARGETS flecsolve
  EXPORT
  flecsolveTargets
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}
  FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flecsolve)

install(EXPORT flecsolveTargets
  FILE
  flecsolveTargets.cmake
  NAMESPACE
  flecsolve::
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/flecsolve)

# version
set_target_properties(flecsolve
  PROPERTIES
  VERSION ${flecsolve_VERSION})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/flecsolveConfigVersion.cmake
  VERSION ${flecsolve_VERSION}
  COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/flecsolveConfigVersion.cmake
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/${target})

# local export
export(EXPORT flecsolveTargets
  FILE ${CMAKE_BINARY_DIR}/flecsolveTargets.cmake
  NAMESPACE flecsolve::)

export(PACKAGE flecsolve)

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
if(FLECSOLVE_ENABLE_FORMAT AND TARGET ClangFormat::ClangFormat)
  flecsi_summary_info("ClangFormat_EXECUTABLE" "${ClangFormat_EXECUTABLE}" TRUE)
endif()
string(APPEND _summary "\n")
flecsi_summary_info("FLECSI_BACKEND" "${FleCSI_BACKEND}" TRUE)
flecsi_summary_option("FLECSI_ENABLE_FLOG" ${FleCSI_ENABLE_FLOG} "")
flecsi_summary_option("FLECSOLVE_ENABLE_UNIT_TESTS" ${FLECSOLVE_ENABLE_UNIT_TESTS} "")
flecsi_summary_option("FLECSOLVE_ENABLE_AMP" "${FLECSOLVE_ENABLE_AMP}" "")
flecsi_summary_option("FLECSOLVE_BUILD_EXAMPLES" ${FLECSOLVE_BUILD_EXAMPLES} "")
flecsi_summary_info("AMP_DATA" "${AMP_DATA}" TRUE)

message(STATUS ${_summary})
