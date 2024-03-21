flecsolve_headers(
  base.hh
  bdf_parameters.hh
  bdf.hh
  parameters.hh
  rk23.hh
  rk45.hh)

flecsolve_sources(
  bdf.cc
  bdf_parameters.cc)

flecsi_add_test(explicit
  SOURCES
  test/explicit.cc
  INPUTS
  test/explicit.cfg
  LIBRARIES flecsolve
  PROCS 1)

flecsi_add_test(implicit
  SOURCES
  test/implicit.cc
  INPUTS
  test/implicit.cfg
  LIBRARIES flecsolve
  PROCS 1)
