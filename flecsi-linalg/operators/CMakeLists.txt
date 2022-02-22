add_unit(cg
  SOURCES
  test/cg.cc
  INPUTS
  ${PROJECT_SOURCE_DIR}/data/494_bus/494_bus.mtx
  ${PROJECT_SOURCE_DIR}/data/Chem97ZtZ/Chem97ZtZ.mtx
  LIBRARIES
  FleCSI::linalg
  THREADS 1)
