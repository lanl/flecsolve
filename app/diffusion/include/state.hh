#pragma once

#include "flecsi-linalg/physics/specializations/operator_mesh.hh"
#include <flecsi/data.hh>

#include "parameters.hh"

using namespace flecsi;

namespace diffusion {

using msh = linalg::physics::operator_mesh;

msh::slot m;
msh::cslot coloring;

const field<scalar_t>::definition<msh, msh::cells> v1d, v2d, rhs1d, rhs2d,
	diffa;
const field<scalar_t>::definition<msh, msh::faces> diffb;

} // namespace diffusion