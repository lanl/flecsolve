set(LIBNAME "operators")

add_library(${LIBNAME} INTERFACE)
add_library(${CMAKE_PROJECT_NAME}::${LIBNAME} ALIAS ${LIBNAME})

set(SOURCE_LIST
  "include/operators/operator_expression.hh"
  "include/operators/operator_task.hh"
  "include/operators/operator_kernels.hh"
  "include/operators/operator_utils.hh"
  "include/operators/boundary_operator/dirichlet.hh"
  "include/operators/boundary_operator/neumann.hh"
  "include/operators/common/operator_base.hh"
  "include/operators/common/state_store.hh"
  "include/operators/volume_diffusion/volume_diffusion.hh"
  "include/operators/specialization/operator_mesh.hh"

)

target_sources(
  ${LIBNAME}
  INTERFACE
    ${SOURCE_LIST}
)

target_include_directories(
  ${LIBNAME}
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
)

target_compile_features(
  ${LIBNAME}
  INTERFACE
    cxx_std_17
)

target_link_libraries(${LIBNAME}
  INTERFACE
  FleCSI::linalg
)

source_group(
  TREE "${CMAKE_CURRENT_LIST_DIR}"
  FILES "${SOURCE_LIST}"
)

# add_unit(operator_task_test
#   SOURCES
#     test/operator_task_test.cc
#   LIBRARIES
#     ${LIBNAME}
#     FleCSI::linalg
#     FleCSI::FleCSI
#     MPI::MPI_CXX
#   THREADS 4
# )

# add_unit(dirichlet_test
#   SOURCES
#     test/dirichlet_test.cc
#   LIBRARIES
#     ${LIBNAME}
#     FleCSI::linalg
#     FleCSI::FleCSI
#     MPI::MPI_CXX
#   THREADS 4
# )

# add_unit(operator_expr_test
#   SOURCES
#     test/operator_expr_test.cc
#   LIBRARIES
#     ${LIBNAME}
#     FleCSI::linalg
#     FleCSI::FleCSI
#     MPI::MPI_CXX
#   THREADS 4
# )

# add_unit(volume_diffusion_test
#   SOURCES
#     test/volume_diffusion_test.cc
#   LIBRARIES
#     ${LIBNAME}
#     FleCSI::linalg
#     FleCSI::FleCSI
#     MPI::MPI_CXX
#   THREADS 4
# )