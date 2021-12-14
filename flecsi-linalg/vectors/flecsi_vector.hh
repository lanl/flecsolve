#pragma once

#include "vector.hh"
#include "operations/flecsi_operations.hh"
#include "data/flecsi_data.hh"

namespace flecsi::linalg {

template <class Topo, typename Topo::index_space Space, class Real = double>
using flecsi_vector = vector<flecsi_data<Topo, Space, Real>,
                             flecsi_operations<Topo, Space, Real>>;
}
