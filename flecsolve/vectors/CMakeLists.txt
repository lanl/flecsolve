set(vectors_HEADERS
  mesh.hh
  seq.hh
  multi.hh
  util.hh
  variable.hh
  vector.hh
  base.hh
  data/mesh.hh
  operations/mesh.hh
  operations/multi.hh
  operations/mesh_tasks.hh
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
