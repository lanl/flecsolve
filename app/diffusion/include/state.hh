#pragma once

#include "flecsolve/physics/specializations/operator_mesh.hh"
#include <flecsi/data.hh>

#include "parameters.hh"

using namespace flecsi;

namespace diffusion {

using msh = flecsolve::physics::operator_mesh;

msh::slot m;
msh::cslot coloring;

const field<scalar_t>::definition<msh, msh::cells> v1d, v2d, rhs1d, rhs2d,
	diffa;
const field<scalar_t>::definition<msh, msh::faces> diffb;

} // namespace diffusion
