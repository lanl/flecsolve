
add_unit(fvm_mesh_test
  SOURCES
    test/fvm_mesh.cc
  LIBRARIES
    FleCSI::linalg
  THREADS
    1
  )
add_unit(boundary_test
  SOURCES
    test/boundaries.cc
  LIBRARIES
    FleCSI::linalg
  THREADS
    1
  )