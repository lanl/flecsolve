set(solvers_HEADERS
  bicgstab.hh
  cg.hh
  gmres.hh
  krylov_operator.hh
  krylov_parameters.hh
  nka.hh
  solver_settings.hh
  traits.hh
  factory.hh
  mg/jacobi.hh
  mg/gs.hh
  PARENT_SCOPE)

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
  test/nka-factory.cfg
  LIBRARIES flecsolve
  PROCS 1)

add_subdirectory(mg)
