#pragma once

#include "vector.hh"
#include "operations/petsc.hh"
#include "data/petsc.hh"

namespace flecsolve::vec {

using petsc = vector<data::petsc, ops::petsc>;

}
