add_unit(cg
  SOURCES
  test/cg.cc
  INPUTS
  ${PROJECT_SOURCE_DIR}/data/494_bus/494_bus.mtx
  ${PROJECT_SOURCE_DIR}/data/Chem97ZtZ/Chem97ZtZ.mtx
  LIBRARIES
  FleCSI::linalg
  THREADS 1)

add_unit(gmres
  SOURCES
  test/gmres.cc
  INPUTS
  ${PROJECT_SOURCE_DIR}/data/Chem97ZtZ/Chem97ZtZ.mtx
  LIBRARIES
  FleCSI::linalg
  THREADS 1)

add_unit(bicgstab
  SOURCES
  test/bicgstab.cc
  INPUTS
  ${PROJECT_SOURCE_DIR}/data/Chem97ZtZ/Chem97ZtZ.mtx
  LIBRARIES
  FleCSI::linalg
  THREADS 1)
