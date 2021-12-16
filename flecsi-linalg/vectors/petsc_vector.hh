#pragma once

#include "vector.hh"
#include "operations/petsc_operations.hh"
#include "data/petsc_data.hh"


namespace flecsi::linalg {

using petsc_vector = vector<petsc_data, petsc_operations>;

}
