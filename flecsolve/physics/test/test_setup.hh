#pragma once

#include <array>

#include <flecsi/flog.hh>
#include <iostream>

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/physics/boundary/dirichlet.hh"
#include "flecsolve/physics/boundary/neumann.hh"
#include "flecsolve/physics/expressions/operator_expression.hh"
#include "flecsolve/physics/volume_diffusion/volume_diffusion.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/solver_settings.hh"

using namespace flecsi;
namespace flecsolve {
namespace physics_testing {

using scalar_t = double;

using msh = physics::operator_mesh;

msh::slot m;
msh::cslot coloring;

inline void init_mesh(const std::vector<std::size_t> & extents) {

	auto colors = msh::distribute(processes(), extents);
	coloring.allocate(colors, extents);

	msh::gbox geometry;
	geometry[msh::x_axis][0] = 0.0;
	geometry[msh::x_axis][1] = 1.0;
	geometry[msh::y_axis] = geometry[msh::x_axis];
	geometry[msh::z_axis] = geometry[msh::x_axis];

	m.allocate(coloring.get(), geometry);
}

inline void check_vals(msh::accessor<ro, ro> vm,
                       field<scalar_t>::accessor<ro, na> xa,
                       std::string title) {
	auto xv = vm.mdspan<msh::cells>(xa);

	std::ostringstream oss;

	oss << "====================\n";
	oss << "--------------------\n";
	oss << title << "\n";
	oss << "--------------------\n";
	for (auto j : vm.range<msh::cells, msh::y_axis, msh::logical>()) {
		oss << "j = " << j << std::setw(6) << " | ";
		for (auto i : vm.range<msh::cells, msh::x_axis, msh::logical>()) {
			oss << xv[1][j][i] << " ";
		}
		oss << "\n";
	}
	oss << "====================\n";

	oss << "\n";
	std ::cout << oss.str();
}

template<auto Space>
inline void fill_field(msh::accessor<ro, ro> vm,
                       field<scalar_t>::accessor<wo, na> xa,
                       scalar_t val)

{
	auto xv = vm.mdspan<Space>(xa);
	for (auto j : vm.range<Space, msh::y_axis, msh::all>()) {
		for (auto i : vm.range<Space, msh::x_axis, msh::all>()) {
			xv[1][j][i] = val;
		}
	}
}

/**
 * @brief initalizes values to slope upwards in 𝒙
 *
 *   │
 *   │        ┌──┐
 *   │     ┌──┤  │
 *   │  ┌──┤  │  │
 *   ├──┤  │  │  │
 *   └──┴──┴──┴──┴───
 *   X───►
 * @param vm mesh accessor (read-only)
 * @param xa field accessor (write-only)
 */
inline void slope_field(msh::accessor<ro, ro> vm,
                        field<scalar_t>::accessor<wo, na> xa) {
	auto xv = vm.mdspan<msh::cells>(xa);

	for (auto j : vm.range<msh::cells, msh::y_axis, msh::logical>()) {
		for (auto i : vm.range<msh::cells, msh::x_axis, msh::logical>()) {
			xv[1][j][i] = vm.value<msh::x_axis>(i);
		}
	}
}

} // namespace physics_testing
} // namespace flecsolve