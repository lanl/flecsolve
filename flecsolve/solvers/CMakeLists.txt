set(solvers_HEADERS
  bicgstab.hh
  cg.hh
  gmres.hh
  krylov_interface.hh
  nka.hh
  solver_settings.hh
  traits.hh
  factory.hh
  PARENT_SCOPE)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()

if(ENABLE_UNIT_TESTS)
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

flecsi_add_test(cg
  SOURCES
  test/cg.cc
  INPUTS
  ${494_bus_SOURCE_DIR}/494_bus.mtx
  ${chem97ztz_SOURCE_DIR}/Chem97ZtZ.mtx
  test/cg.cfg
  LIBRARIES flecsolve
  PROCS 1)

flecsi_add_test(cgmulti
  SOURCES
  test/cgmulti.cc
  INPUTS
  ${chem97ztz_SOURCE_DIR}/Chem97ZtZ.mtx
  test/cgmulti.cfg
  LIBRARIES flecsolve
  PROCS 1)

flecsi_add_test(gmres
  SOURCES
  test/gmres.cc
  INPUTS
  ${chem97ztz_SOURCE_DIR}/Chem97ZtZ.mtx
  test/gmres.cfg
  LIBRARIES flecsolve
  PROCS 1)

flecsi_add_test(bicgstab
  SOURCES
  test/bicgstab.cc
  INPUTS
  ${chem97ztz_SOURCE_DIR}/Chem97ZtZ.mtx
  ${psmigr_3_SOURCE_DIR}/psmigr_3.mtx
  test/bicgstab.cfg
  LIBRARIES flecsolve
  PROCS 1)

flecsi_add_test(nka
  SOURCES
  test/nka.cc
  INPUTS
  ${chem97ztz_SOURCE_DIR}/Chem97ZtZ.mtx
  test/nka.cfg
  LIBRARIES flecsolve
  PROCS 1)
