#pragma once

#include <array>

#include <flecsi/flog.hh>
#include <flecsi/util/constant.hh>
#include <iomanip>
#include <iostream>

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/util/config.hh"
#include "flecsolve/physics/boundary/dirichlet.hh"
#include "flecsolve/physics/boundary/neumann.hh"
#include "flecsolve/physics/expressions/operator_expression.hh"
#include "flecsolve/physics/volume_diffusion/diffusion.hh"
#include "flecsolve/physics/volume_diffusion/coefficient.hh"
#include "flecsolve/solvers/krylov_interface.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/gmres.hh"
#include "flecsolve/solvers/nka.hh"
#include "flecsolve/solvers/solver_settings.hh"

#include "parameters.hh"
#include "state.hh"

using namespace flecsi;
namespace eqdiff {
void init_mesh() {
	std::vector<std::size_t> extents{{NX, NY, 1}};
	auto colors = msh::distribute(processes(), extents);
	coloring.allocate(colors, extents);

	msh::gbox geometry;
	geometry[msh::x_axis][0] = 0.0;
	geometry[msh::x_axis][1] = 1.0;
	geometry[msh::y_axis] = geometry[msh::x_axis];
	geometry[msh::z_axis] = geometry[msh::x_axis];

	m.allocate(coloring.get(), geometry);
}

void check_vals(msh::accessor<ro, ro> vm,
                field<scalar_t>::accessor<ro, na> xa,
                std::string title) {
	auto xv = vm.mdspan<msh::cells>(xa);

	std::ostringstream oss;

	oss << "====================\n";
	oss << "--------------------\n";
	oss << title << "\n";
	oss << "--------------------\n";
	oss << std::fixed << std::setprecision(6) << std::setfill(' ');
	for (auto j : vm.range<msh::cells, msh::y_axis, msh::all>()) {
		oss << "j = " << j << std::setw(3) << " | ";
		for (auto i : vm.range<msh::cells, msh::x_axis, msh::all>()) {
			oss << std::setw(9) << xv[1][j][i] << " ";
		}
		oss << "\n";
	}
	oss << "====================\n";

	oss << "\n";
	std ::cout << oss.str();
}

template<auto Space>
void fill_field(msh::accessor<ro, ro> vm,
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
void slope_field(msh::accessor<ro, ro> vm,
                 field<scalar_t>::accessor<wo, na> xa) {
	auto xv = vm.mdspan<msh::cells>(xa);

	for (auto k : vm.range<msh::cells, msh::z_axis, msh::extended>()) {
		for (auto j : vm.range<msh::cells, msh::y_axis, msh::extended>()) {
			for (auto i : vm.range<msh::cells, msh::x_axis, msh::extended>()) {
				xv[k][j][i] = vm.value<msh::x_axis>(i + 1);
			}
		}
	}
}

template<class Vec>
constexpr decltype(auto) make_boundary_operator_neumann(const Vec &) {
	using namespace flecsolve::physics;

	auto bndxl = bc<neumann<Vec>, msh::x_axis, msh::boundary_low>::create({});
	auto bndxh = bc<neumann<Vec>, msh::x_axis, msh::boundary_high>::create({});
	auto bndyl = bc<neumann<Vec>, msh::y_axis, msh::boundary_low>::create({});
	auto bndyh = bc<neumann<Vec>, msh::y_axis, msh::boundary_high>::create({});

	return op_expr(
		flecsolve::multivariable<Vec::var.value>, bndxl, bndxh, bndyl, bndyh);
}

template<class Vec>
constexpr decltype(auto) make_boundary_operator_dirichlet(const Vec &) {
	using namespace flecsolve::physics;

	auto bndxl = bc<dirichlet<Vec>, msh::x_axis, msh::boundary_low>::create({1.0E-9});
	auto bndxh = bc<dirichlet<Vec>, msh::x_axis, msh::boundary_high>::create({1.0E-9});
	auto bndyl = bc<dirichlet<Vec>, msh::y_axis, msh::boundary_low>::create({1.0E-9});
	auto bndyh = bc<dirichlet<Vec>, msh::y_axis, msh::boundary_high>::create({1.0E-9});
	return op_expr(
		flecsolve::multivariable<Vec::var.value>, bndxl, bndxh, bndyl, bndyh);
}

template<class Vec>
constexpr decltype(auto) make_boundary_operator_pseudo(const Vec &) {
	using namespace flecsolve::physics;

	auto bndl = bc<neumann<Vec>, msh::z_axis, msh::boundary_low>::create({});
	auto bndh = bc<neumann<Vec>, msh::z_axis, msh::boundary_high>::create({});
	return op_expr(flecsolve::multivariable<Vec::var.value>, bndl, bndh);
}

template<auto N, class Vec>
decltype(auto) make_volume_operator(const Vec & v) {
	using namespace flecsolve::physics;

	flecsi::util::key_array<flecsi::field<scalar_t>::Reference<msh, msh::faces>,
	                        msh::axes>
		bref{diffb[N][msh::x_axis](m),
	         diffb[N][msh::y_axis](m),
	         diffb[N][msh::z_axis](m)};

	//auto vd = operator_creator<diffusion<Vec>, constant_coefficent>::create(bref, diffa[N](m), 1.0, 0.0, m);
	auto coeffop = constant_coefficent<Vec>::create({bref});
	auto voldiff =
		diffusion<Vec>::create({diffa[N](m), bref, 1.0, 0.0}, m);
	return op_expr(flecsolve::multivariable<Vec::var.value>, coeffop, voldiff);
}

int driver() {

	// initialize the mesh
	init_mesh();

	// fill auxiliary data fields
	for (std::size_t n = 0; n < NVAR; ++n) {
		execute<fill_field<msh::cells>>(m, diffa[n](m), DEFAULT_VAL);
	}

	// set up initial conditions
	execute<slope_field>(m, vd[0](m));
	execute<slope_field>(m, vd[1](m));

	if (processes() < 100) {
		execute<check_vals>(m, vd[0](m), "vd0");
		//"[variable1] solution with zero-flux boundary (n ⋅ ∇ u=0 on ∂Ω) ");
		execute<check_vals>(m, vd[1](m), "vd1");
		//"[variable2] solution vanishes at boundary (u=0 on ∂Ω)");
	}
	else {
		flog(info) << "to see asci representation of ivs & solutions, run on a "
					  "single core.\n";
	}
	//===================================================
	//===============multivector diffusion===============
	//===================================================

	// define the solution and RHS MVs and assign them to a variable/field
	flecsolve::vec::multi X(
		flecsolve::vec::mesh(
			flecsolve::variable<diffusion_var::v1>, m, vd[0](m)),
		flecsolve::vec::mesh(
			flecsolve::variable<diffusion_var::v2>, m, vd[1](m)));

	flecsolve::vec::multi RHS(
		flecsolve::vec::mesh(
			flecsolve::variable<diffusion_var::v1>, m, rhsd[0](m)),
		flecsolve::vec::mesh(
			flecsolve::variable<diffusion_var::v2>, m, rhsd[1](m)));

	auto & [vec1, vec2] = X;

	auto bnd_op_1 =
		flecsolve::physics::op_expr(flecsolve::multivariable<diffusion_var::v1>,
	                                make_boundary_operator_dirichlet(vec1),
	                                make_boundary_operator_pseudo(vec1));
	auto bnd_op_2 =
		flecsolve::physics::op_expr(flecsolve::multivariable<diffusion_var::v2>,
	                                make_boundary_operator_neumann(vec2),
	                                make_boundary_operator_pseudo(vec2));
	// build the operator on the variables
	auto A = flecsolve::physics::op_expr(
		flecsolve::multivariable<diffusion_var::v1, diffusion_var::v2>,
		bnd_op_1,
		make_volume_operator<0>(vec1),
		bnd_op_2,
		make_volume_operator<1>(vec2));

	std :: cout << A.to_string() << std::endl;

	auto apar = A.get_parameters<0>(X);

	A.reset(apar);
	// set the RHS to vanish
	RHS.set_scalar(0.0);


	// get the solver parameters and workspace, & bind the operator to the
	// solver
	flecsolve::op::krylov_parameters params(
		flecsolve::cg::settings("solver"),
		flecsolve::cg::topo_work<>::get(RHS),
		std::ref(A));
	// flecsolve::op::krylov_parameters params(
	// 	flecsolve::gmres::settings("solver"),
	// 	flecsolve::gmres::topo_work<>::get(RHS),
	// 	std::ref(A));
	read_config("diffusion.cfg", params);

	// create the solver
	flecsolve::op::krylov slv(std::move(params));

	// run the solver
	auto info = slv.apply(RHS, X);

	// print some statistics on the solve
	flog(info) << "norm = " << info.res_norm_final << "\n";
	flog(info) << "iters = " << info.iters << "\n";

	// helper print of final solutions
	if (processes() < 100) {
		execute<check_vals>(m, vd[0](m), "vd0");
		//"[variable1] solution with zero-flux boundary (n ⋅ ∇ u=0 on ∂Ω) ");
		execute<check_vals>(m, vd[1](m), "vd1");
		//"[variable2] solution vanishes at boundary (u=0 on ∂Ω)");
	}
	else {
		flog(info) << "to see asci representation of ivs & solutions, run on a "
					  "single core.\n";
	}

	return 0;
}
} // namespace diffusion
