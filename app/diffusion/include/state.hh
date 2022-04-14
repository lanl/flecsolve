#pragma once

#include "flecsi-linalg/discrete_operators/specializations/operator_mesh.hh"
#include <flecsi/data.hh>

#include "parameters.hh"

using namespace flecsi;

namespace diffusion {

using msh = linalg::discrete_operators::operator_mesh;

msh::slot m;
msh::cslot coloring;

field<scalar_t>::definition<msh, msh::cells> v1d, v2d, rhs1d, rhs2d, diffa;
field<scalar_t>::definition<msh, msh::faces> diffb;

} // namespace diffusion