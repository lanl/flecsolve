set(matrices_HEADERS
  par.hh
  seq.hh
  PARENT_SCOPE)

flecsi_add_test(parcsr
  SOURCES
  test/par_csr.cc
  LIBRARIES flecsolve
  INPUTS
  test/parcsr.cfg
  ${chem97ztz_SOURCE_DIR}/Chem97ZtZ.mtx
  PROCS 3)
