flecsi_add_test(jacobi
  SOURCES
  test/jacobi.cc
  LIBRARIES flecsolve
  PROCS 1)

flecsi_add_test(coarsen
  SOURCES
  test/coarsen.cc
  LIBRARIES flecsolve
  PROCS 1)
