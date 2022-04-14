
add_unit(boundary_test
  SOURCES
    test/boundaries.cc
  LIBRARIES
    FleCSI::linalg
  THREADS
    4
  )


add_unit(multivectorop_test
SOURCES
  test/multivector_operator.cc
LIBRARIES
  FleCSI::linalg
THREADS
  4
)
