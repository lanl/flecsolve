flecsolve_headers(
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
  mg/coarsen.hh
  mg/intergrid.hh)

flecsolve_sources(
  amp.cc)

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

if(TPL_FOUND_LIBMESH AND AMP_DATA)
  flecsi_add_test(amp-amg
	SOURCES
	test/amp_boomeramg.cc
	INPUTS
	${AMP_DATA}/meshes/CubeMeshes/cube64.with.boundary.labels.e
	test/amp-amg.cfg
	test/amp-input
	LIBRARIES flecsolve
	PROCS 3)

  flecsi_add_test(amp-solver
	SOURCES
	test/amp_solver.cc
	INPUTS
	${AMP_DATA}/meshes/CubeMeshes/cube64.with.boundary.labels.e
	test/amp-solver.cfg
	test/amp-solver-pcg.cfg
	test/amp-solver-input
	LIBRARIES flecsolve
	PROCS 3)
endif()

add_subdirectory(mg)
