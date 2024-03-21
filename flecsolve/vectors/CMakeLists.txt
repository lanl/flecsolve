flecsolve_headers(
  topo_view.hh
  seq.hh
  core.hh
  multi.hh
  util.hh
  variable.hh
  traits.hh
  data/topo_view.hh
  data/multi.hh
  operations/topo_view.hh
  operations/multi.hh
  operations/topo_tasks.hh)

flecsolve_sources(
  util.cc
)

flecsi_add_test(flecsi_vector
  SOURCES
  test/flecsi_vector.cc
  LIBRARIES flecsolve
  PROCS 4)

flecsi_add_test(flecsi_multivector
  SOURCES
  test/flecsi_multivector.cc
  LIBRARIES flecsolve
  PROCS 4)
