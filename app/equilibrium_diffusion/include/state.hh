#pragma once

#include "flecsolve/physics/specializations/operator_mesh.hh"
#include <flecsi/data.hh>
#include <flecsi/util/constant.hh>

#include "parameters.hh"

using namespace flecsi;

namespace eqdiff {

using msh = flecsolve::physics::operator_mesh;

msh::slot m;
msh::cslot coloring;

std::array<field<scalar_t>::definition<msh, msh::cells>, NVAR> vd, rhsd, diffa;

std::array<
	util::key_array<field<scalar_t>::definition<msh, msh::faces>, msh::axes>,
	NVAR>
	diffb;

template<auto N>
constexpr inline auto faces_ref()
{
	return
		util::key_array<flecsi::field<scalar_t>::Reference<msh, msh::faces>, msh::axes>
			{diffb[N][msh::x_axis](m),
			diffb[N][msh::y_axis](m),
			diffb[N][msh::z_axis](m)};
}

} // namespace diffusion
