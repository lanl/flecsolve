flecsi_add_test(jacobi
  SOURCES
  test/jacobi.cc
  LIBRARIES flecsolve
  PROCS 1)

flecsi_add_test(galerkin_prod
  SOURCES
  test/galerkin_prod.cc
  LIBRARIES flecsolve
  INPUTS
  ${nos7_SOURCE_DIR}/nos7.mtx
  PROCS 2)

flecsi_add_test(intergrid
  SOURCES
  test/intergrid.cc
  LIBRARIES flecsolve
  INPUTS
  ${nos7_SOURCE_DIR}/nos7.mtx
  PROCS 3)
