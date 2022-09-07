
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

add_unit(fvm_coeff_test
SOURCES
  test/fvm_coeff.cc
LIBRARIES
  FleCSI::linalg
THREADS
  1
)

add_unit(fvm_diffusion_test
SOURCES
  test/fvm_diffusion.cc
LIBRARIES
  FleCSI::linalg
THREADS
  1
)