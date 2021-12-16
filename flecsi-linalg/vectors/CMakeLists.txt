add_unit(flecsi_vector
  SOURCES
  test/flecsi_vector.cc
  LIBRARIES
  FleCSI::linalg
  THREADS 4)

add_unit(flecsi_multivector
  SOURCES
  test/flecsi_multivector.cc
  LIBRARIES
  FleCSI::linalg
  THREADS 4)

add_unit(petsc_interop
  SOURCES
  test/petsc.cc
  LIBRARIES
  FleCSI::linalg
  THREADS 4)
