set(physics_HEADERS
  boundary/bc_base.hh
  boundary/dirichlet.hh
  boundary/neumann.hh
  boundary/robin.hh
  common/operator_base.hh
  common/operator_utils.hh
  common/state_store.hh
  common/vector_types.hh
  expressions/operator_expression.hh
  reaction/arrhenius.hh
  reaction/fkn_mechanism.hh
  specializations/fvm_narray.hh
  volume_diffusion/diffusion.hh
  volume_diffusion/coefficient.hh
  PARENT_SCOPE)

add_unit(fvm_mesh_test
  SOURCES
    test/fvm_mesh.cc
  LIBRARIES
    flecsolve
  THREADS
    1
  )

add_unit(boundary_test
  SOURCES
  test/boundaries.cc
  LIBRARIES flecsolve
  THREADS
    1
  )

add_unit(fvm_coeff_test
SOURCES
  test/fvm_coeff.cc
LIBRARIES
  flecsolve
THREADS
  1
)

add_unit(fvm_diffusion_test
SOURCES
  test/fvm_diffusion.cc
LIBRARIES
  flecsolve
THREADS
  1
)
