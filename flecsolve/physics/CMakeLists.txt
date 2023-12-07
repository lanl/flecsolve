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

flecsi_add_test(
  fvm_mesh_test
  SOURCES
  test/fvm_mesh.cc
  LIBRARIES
  flecsolve
  PROCS
  1)

# flecsi_add_test(boundary_test SOURCES test/boundaries.cc LIBRARIES flecsolve
# PROCS 1 )
#
# flecsi_add_test(fvm_coeff_test SOURCES test/fvm_coeff.cc LIBRARIES flecsolve
# PROCS 1 )
#
# flecsi_add_test(fvm_diffusion_test SOURCES test/fvm_diffusion.cc LIBRARIES
# flecsolve PROCS 1 )
