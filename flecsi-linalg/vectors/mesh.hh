#pragma once

#include "vector.hh"
#include "operations/mesh.hh"
#include "data/mesh.hh"

namespace flecsi::linalg::vec {

template <class Topo, typename Topo::index_space Space, class Real = double>
using mesh = vector<data::mesh<Topo, Space, Real>,
                    ops::mesh<Topo, Space, Real>>;

}
