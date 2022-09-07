#pragma once

#include <array>

#include <flecsi/flog.hh>
#include <iostream>

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/physics/specializations/fvm_narray.hh"
#include "flecsolve/physics/boundary/dirichlet.hh"
#include "flecsolve/physics/boundary/neumann.hh"
#include "flecsolve/physics/expressions/operator_expression.hh"
#include "flecsolve/physics/volume_diffusion/diffusion.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/solver_settings.hh"

using namespace flecsi;
namespace flecsolve {
namespace physics_testing {

using scalar_t = double;

using msh = physics::fvm_narray;

msh::slot m;
msh::cslot coloring;

template<auto S>
using fld = const field<scalar_t>::definition<msh, S>;

template<auto S>
using fldr = flecsi::field<scalar_t>::Reference<msh, S>;

constexpr scalar_t DEFAULT_TOL = 1.0E-8;

const inline auto
make_faces_ref(const util::key_array<fld<msh::faces>, msh::axes> & fs) {
	return util::key_array<fldr<msh::faces>, msh::axes>{
		fs[msh::x_axis](m), fs[msh::y_axis](m), fs[msh::z_axis](m)};
}

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
	for (auto j : vm.range<msh::cells, msh::y_axis, msh::all>()) {
		oss << "j = " << j << std::setw(6) << " | ";
		for (auto i : vm.range<msh::cells, msh::x_axis, msh::all>()) {
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
	for (auto k : vm.range<Space, msh::z_axis, msh::all>()) {
		for (auto j : vm.range<Space, msh::y_axis, msh::all>()) {
			for (auto i : vm.range<Space, msh::x_axis, msh::all>()) {
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

	for (auto j : vm.range<msh::cells, msh::y_axis, msh::logical>()) {
		for (auto i : vm.range<msh::cells, msh::x_axis, msh::logical>()) {
			xv[1][j][i] = vm.value<msh::x_axis>(i);
		}
	}
}

template<auto x>
using fconstant = std::integral_constant<std::decay_t<decltype(x)>, x>;

template<class F, class T, class A, class S, class D>
struct fvm_check {
	F f;
	T name;
	A ax;
	S sp;
	D dm;
	int operator()(msh::accessor<ro, ro> vm,
	               field<double>::accessor<ro, na> x) {

		UNIT (name) {
			auto xv = vm.mdspan<S::value>(x);
			auto [kk, jj, ii] = vm.full_range<S::value, A::value, D::value>();

			for (auto k : kk) {
				for (auto j : jj) {
					for (auto i : ii) {
						EXPECT_LT(std::abs(f(k, j, i) - xv[k][j][i]),
						          DEFAULT_TOL);
					}
				}
			}
		};
	}
};

template<class F, class T, class A, class S, class D>
fvm_check(F &&, T &&, A &&, S &&, D &&) -> fvm_check<F, T, A, S, D>;

template<class F, class T, class A, class D>
fvm_check(F &&, T &&, A &&, D &&)
	-> fvm_check<F, T, A, fconstant<msh::cells>, D>;

template<class F, class T>
fvm_check(F &&, T &&) -> fvm_check<F,
                                   T,
                                   fconstant<msh::x_axis>,
                                   fconstant<msh::cells>,
                                   fconstant<msh::logical>>;

} // namespace physics_testing
} // namespace flecsolve