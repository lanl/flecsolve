/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
#pragma once

#include <array>

#include <flecsi/flog.hh>
#include <iostream>

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/physics/specializations/fvm_narray.hh"

using namespace flecsi;
namespace flecsolve {
namespace physics_testing {

using scalar_t = double;

using msh = physics::fvm_narray;

template<auto S>
using fld = const field<scalar_t>::definition<msh, S>;

template<auto S>
using fldr = flecsi::field<scalar_t>::Reference<msh, S>;

constexpr scalar_t DEFAULT_TOL = 1.0E-8;

inline auto
make_faces_ref(msh::slot & m,
               const util::key_array<fld<msh::faces>, msh::axes> & fs) {
	return util::key_array<fldr<msh::faces>, msh::axes>{
		{fs[msh::x_axis](m), fs[msh::y_axis](m), fs[msh::z_axis](m)}};
}

inline void init_mesh(msh::slot & m, const std::vector<util::gid> & extents) {
	using mesh = physics::fvm_narray;
	using mbase = mesh::base;
	mesh::index_definition idef, idef_faces;
	idef.axes = mbase::make_axes(
		mbase::distribute(flecsi::processes(), extents), extents);
	for (auto & a : idef.axes) {
		a.hdepth = 1;
		a.bdepth = 1;
	}

	idef_faces.axes = mbase::make_axes(
		mbase::distribute(flecsi::processes(), extents), extents);

	for (auto & a : idef_faces.axes) {
		a.hdepth = 0;
		a.bdepth = 0;
		a.auxiliary = true;
	}

	msh::gbox geometry;
	geometry[msh::x_axis][0] = 0.0;
	geometry[msh::x_axis][1] = 1.0;
	geometry[msh::y_axis] = geometry[msh::x_axis];
	geometry[msh::z_axis] = geometry[msh::x_axis];

	m.allocate(mesh::mpi_coloring(idef, idef_faces), geometry);
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
	for (auto j : vm.range<msh::cells, msh::y_axis, msh::domain::all>()) {
		oss << "j = " << j << std::setw(6) << " | ";
		for (auto i : vm.range<msh::cells, msh::x_axis, msh::domain::all>()) {
			oss << xv[1][j][i] << " ";
		}
		oss << "\n";
	}
	oss << "====================\n";

	oss << "\n";
	std ::cout << oss.str();

	for (auto i : vm.dofs<msh::cells>()) {
		std ::cout << i << " ";
	}
	std::cout << "\n";
}

template<auto Space>
inline void fill_field(msh::accessor<ro, ro> vm,
                       field<scalar_t>::accessor<wo, na> xa,
                       scalar_t val)

{
	auto xv = vm.mdspan<Space>(xa);
	for (auto k : vm.range<Space, msh::z_axis, msh::domain::all>()) {
		for (auto j : vm.range<Space, msh::y_axis, msh::domain::all>()) {
			for (auto i : vm.range<Space, msh::x_axis, msh::domain::all>()) {
				xv[k][j][i] = val;
			}
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

	for (auto j : vm.range<msh::cells, msh::y_axis, msh::domain::logical>()) {
		for (auto i :
		     vm.range<msh::cells, msh::x_axis, msh::domain::logical>()) {
			xv[1][j][i] = vm.value<msh::x_axis>(i);
		}
	}
}

template<auto x>
using fconstant = std::integral_constant<std::decay_t<decltype(x)>, x>;

template<class RT, class FT>
int fvm_check_f(const FT & ft,
                msh::accessor<ro, ro> vm,
                field<double>::accessor<ro, na> x) {
	auto fn = std::get<0>(ft);
	auto name = std::get<1>(ft);
	UNIT (name) {
		auto xv = vm.mdspan<RT::sp>(x);
		auto [kk, jj, ii] = vm.full_range<RT::sp, RT::ax, RT::dm>();

		for (auto k : kk) {
			for (auto j : jj) {
				for (auto i : ii) {
					EXPECT_LT(std::abs(fn(k, j, i) - xv[k][j][i]), DEFAULT_TOL);
				}
			}
		}
	};
}

template<class RT, class FT, class... Args>
auto fvm_run(FT && ft, Args &&... args) {
	return (test<fvm_check_f<RT, FT>, flecsi::mpi>(ft, args...) == 0);
}

// some topo specifications
template<auto A, auto S, auto D>
struct ASD {
	static constexpr auto ax = A;
	static constexpr auto sp = S;
	static constexpr auto dm = D;
};

using rxcl = ASD<msh::x_axis, msh::cells, msh::domain::logical>;

template<auto A, auto D>
using r_c_ = ASD<A, msh::cells, D>;

template<auto A>
using r_lo = r_c_<A, msh::domain::boundary_low>;

template<auto A>
using r_hi = r_c_<A, msh::domain::boundary_high>;

template<auto A>
using rface = ASD<A, msh::faces, msh::domain::logical>;

} // namespace physics_testing
} // namespace flecsolve
