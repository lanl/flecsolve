set(time-integrators_HEADERS
  base.hh
  bdf.hh
  parameters.hh
  rk23.hh
  rk45.hh
  PARENT_SCOPE)

set(time-integrators_SOURCES
  bdf.cc
  PARENT_SCOPE)

add_unit(explicit
  SOURCES
  test/explicit.cc
  INPUTS
  test/explicit.cfg
  LIBRARIES flecsolve
  THREADS 1)

add_unit(implicit
  SOURCES
  test/implicit.cc
  INPUTS
  test/implicit.cfg
  LIBRARIES flecsolve
  THREADS 1)
