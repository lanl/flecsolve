flecsolve_headers(
  parcsr.hh
  seq.hh
  io/matrix_market.hh)

flecsi_add_test(parcsr
  SOURCES
  test/par_csr.cc
  LIBRARIES flecsolve
  INPUTS
  test/parcsr.cfg
  ${chem97ztz_SOURCE_DIR}/Chem97ZtZ.mtx
  ARGUMENTS "--backend-args=-lg:no_tracing"
  PROCS 3)
