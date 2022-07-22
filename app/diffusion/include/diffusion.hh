#pragma once

#include <array>

#include <flecsi/flog.hh>
#include <iostream>

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/util/config.hh"
#include "flecsolve/physics/boundary/dirichlet.hh"
#include "flecsolve/physics/boundary/neumann.hh"
#include "flecsolve/physics/expressions/operator_expression.hh"
#include "flecsolve/physics/volume_diffusion/volume_diffusion.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/solver_settings.hh"

#include "parameters.hh"
#include "state.hh"

using namespace flecsi;
namespace diffusion {
void init_mesh() {
	std::vector<std::size_t> extents{{NX, NY}};
	auto colors = msh::distribute(processes(), extents);
	coloring.allocate(colors, extents);

	msh::grect geometry;
	geometry[0][0] = 0.0;
	geometry[0][1] = 1.0;
	geometry[1] = geometry[0];

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
	for (auto j : vm.range<msh::cells, msh::y_axis, msh::logical>()) {
		oss << "j = " << j << std::setw(6) << " | ";
		for (auto i : vm.range<msh::cells, msh::x_axis, msh::logical>()) {
			oss << xv[j][i] << " ";
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
	for (auto j : vm.range<Space, msh::y_axis, msh::all>()) {
		for (auto i : vm.range<Space, msh::x_axis, msh::all>()) {
			xv[j][i] = val;
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

	for (auto j : vm.range<msh::cells, msh::y_axis, msh::logical>()) {
		for (auto i : vm.range<msh::cells, msh::x_axis, msh::logical>()) {
			xv[j][i] = vm.value<msh::x_axis>(i);
		}
	}
}

template<class Vec>
constexpr decltype(auto) make_boundary_operator_neumann(const Vec &) {
	using namespace flecsolve::physics;

	auto bndxl = make_operator<
		neumann<Vec::var.value, msh, msh::x_axis, msh::boundary_low>>(diffb(m));
	auto bndxh = make_operator<
		neumann<Vec::var.value, msh, msh::x_axis, msh::boundary_high>>(
		diffb(m));
	auto bndyl = make_operator<
		neumann<Vec::var.value, msh, msh::y_axis, msh::boundary_low>>(diffb(m));
	auto bndyh = make_operator<
		neumann<Vec::var.value, msh, msh::y_axis, msh::boundary_high>>(
		diffb(m));

	return op_expr(
		flecsolve::multivariable<Vec::var.value>, bndxl, bndxh, bndyl, bndyh);
}

template<class Vec>
constexpr decltype(auto) make_boundary_operator_dirichlet(const Vec &) {
	using namespace flecsolve::physics;

	auto bndxl = make_operator<
		dirichlet<Vec::var.value, msh, msh::x_axis, msh::boundary_low>>(0.0);
	auto bndxh = make_operator<
		dirichlet<Vec::var.value, msh, msh::x_axis, msh::boundary_high>>(0.0);
	auto bndyl = make_operator<
		dirichlet<Vec::var.value, msh, msh::y_axis, msh::boundary_low>>(0.0);
	auto bndyh = make_operator<
		dirichlet<Vec::var.value, msh, msh::y_axis, msh::boundary_high>>(0.0);

	return op_expr(
		flecsolve::multivariable<Vec::var.value>, bndxl, bndxh, bndyl, bndyh);
}

template<class Vec>
constexpr decltype(auto) make_volume_operator(const Vec &) {
	using namespace flecsolve::physics;

	volume_diffusion_op<Vec::var.value, msh> voldiff(
		m, {diff_beta, diff_alpha, diffa(m), diffb(m)});

	return op_expr(flecsolve::multivariable<Vec::var.value>, voldiff);
}

int driver() {

	// initialize the mesh
	init_mesh();

	// fill auxiliary data fields
	execute<fill_field<msh::cells>>(m, diffa(m), DEFAULT_VAL);
	execute<fill_field<msh::faces>>(m, diffb(m), DEFAULT_VAL);

	// set up initial conditions
	execute<slope_field>(m, v1d(m));
	execute<slope_field>(m, v2d(m));

	// helper output of ICs
	if (processes() == 1) {
		execute<check_vals>(m, v1d(m), "[variable1] initial field");
		execute<check_vals>(m, v2d(m), "[variable2] initial field");
	}

	//===================================================
	//===============multivector diffusion===============
	//===================================================

	// define the solution and RHS MVs and assign them to a variable/field
	flecsolve::vec::multi X(
		flecsolve::vec::mesh(flecsolve::variable<diffusion_var::v1>, m, v1d(m)),
		flecsolve::vec::mesh(
			flecsolve::variable<diffusion_var::v2>, m, v2d(m)));

	flecsolve::vec::multi RHS(
		flecsolve::vec::mesh(
			flecsolve::variable<diffusion_var::v1>, m, rhs1d(m)),
		flecsolve::vec::mesh(
			flecsolve::variable<diffusion_var::v2>, m, rhs2d(m)));

	auto & [vec1, vec2] = X;

	// build the operator on the variables
	auto A = flecsolve::physics::op_expr(
		flecsolve::multivariable<diffusion_var::v1, diffusion_var::v2>,
		make_boundary_operator_neumann(vec1),
		make_volume_operator(vec1),
		make_boundary_operator_dirichlet(vec2),
		make_volume_operator(vec2));

	// set the RHS to vanish
	RHS.set_scalar(0.0);

	// get the solver parameters and workspace, & bind the operator to the
	// solver
	flecsolve::op::krylov_parameters params(
		flecsolve::cg::settings("solver"),
		flecsolve::cg::topo_work<>::get(RHS),
		std::move(A));
	read_config("diffusion.cfg", params);

	// create the solver
	flecsolve::op::krylov slv(std::move(params));

	// run the solver
	auto info = slv.apply(RHS, X);

	// print some statistics on the solve
	flog(info) << "norm = " << info.res_norm_final << "\n";
	flog(info) << "iters = " << info.iters << "\n";

	// helper print of final solutions
	if (processes() == 1) {
		execute<check_vals>(
			m,
			v1d(m),
			"[variable1] solution with zero-flux boundary (n ⋅ ∇ u=0 on ∂Ω) ");
		execute<check_vals>(
			m, v2d(m), "[variable2] solution vanishes at boundary (u=0 on ∂Ω)");
	}
	else {
		flog(info) << "to see asci representation of ivs & solutions, run on a "
					  "single core.\n";
	}

	return 0;
}
} // namespace diffusion
