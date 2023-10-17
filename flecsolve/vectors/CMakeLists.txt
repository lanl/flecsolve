set(vectors_HEADERS
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
  operations/topo_tasks.hh
  PARENT_SCOPE)

set(vectors_SOURCES
  util.cc
  PARENT_SCOPE)

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

if (USE_PETSC)
flecsi_add_test(petsc_interop
  SOURCES
  test/petsc.cc
  PROCS 4)
endif()
