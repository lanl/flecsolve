#pragma once

#include "flecsolve/physics/specializations/operator_mesh.hh"
#include <flecsi/data.hh>
#include <flecsi/util/constant.hh>

#include "parameters.hh"

using namespace flecsi;

namespace diffusion {

using msh = flecsolve::physics::operator_mesh;

msh::slot m;
msh::cslot coloring;

std::array<field<scalar_t>::definition<msh, msh::cells>, NVAR> vd, rhsd, diffa;

std::array<
	util::key_array<field<scalar_t>::definition<msh, msh::faces>, msh::axes>,
	NVAR>
	diffb;

} // namespace diffusion
