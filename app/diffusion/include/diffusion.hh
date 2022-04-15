#pragma once

#include <array>

#include <flecsi/flog.hh>
#include <iostream>

#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/vectors/multi.hh"

#include "flecsi-linalg/discrete_operators/boundary/dirichlet.hh"
#include "flecsi-linalg/discrete_operators/boundary/neumann.hh"
#include "flecsi-linalg/discrete_operators/expressions/operator_expression.hh"
#include "flecsi-linalg/discrete_operators/volume_diffusion/volume_diffusion.hh"
#include "flecsi-linalg/operators/cg.hh"
#include "flecsi-linalg/operators/solver_settings.hh"

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

template<auto Space>
void check_vals(msh::accessor<ro, ro> vm,
                field<scalar_t>::accessor<ro, na> xa) {
	auto xv = vm.mdspan<Space>(xa);

	std::ostringstream oss;

	oss << "[" << flecsi::process() << "] \n";
	for (auto j : vm.range<Space, msh::y_axis, msh::all>()) {
		oss << "j = " << j << std::setw(4) << " | ";
		for (auto i : vm.range<Space, msh::x_axis, msh::all>()) {
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

void slope_field(msh::accessor<ro, ro> vm,
                 field<scalar_t>::accessor<wo, na> xa) {
	auto xv = vm.mdspan<msh::cells>(xa);

	for (auto j : vm.range<msh::cells, msh::y_axis, msh::logical>()) {
		for (auto i : vm.range<msh::cells, msh::x_axis, msh::logical>()) {
			// xv[j][i] = dis(gen);
			xv[j][i] = vm.value<msh::x_axis>(i);
		}
	}
}

template<class Vec>
constexpr decltype(auto) make_boundary_operator_neumann(const Vec & v) {
	using namespace linalg::discrete_operators;
	// using Var = typename std::decay_t<decltype(Vec::var)>;

	auto bndxl =
		make_operator<neumann<Vec::var, msh, msh::x_axis, msh::boundary_low>>(
			diffb(m));
	auto bndxh =
		make_operator<neumann<Vec::var, msh, msh::x_axis, msh::boundary_high>>(
			diffb(m));
	auto bndyl =
		make_operator<neumann<Vec::var, msh, msh::y_axis, msh::boundary_low>>(
			diffb(m));
	auto bndyh =
		make_operator<neumann<Vec::var, msh, msh::y_axis, msh::boundary_high>>(
			diffb(m));

	return op_expr(bndxl, bndxh, bndyl, bndyh);
}

template<class Vec>
constexpr decltype(auto) make_volume_operator(const Vec & v) {
	using namespace linalg::discrete_operators;

	volume_diffusion_op<Vec::var, msh> voldiff(
		m, {diff_beta, diff_alpha, diffa(m), diffb(m)});

	return op_expr(voldiff);
}

//#define _RUN_MULTI

int driver() {

	init_mesh();

	execute<fill_field<msh::cells>>(m, diffa(m), DEFAULT_VAL);
	execute<fill_field<msh::faces>>(m, diffb(m), DEFAULT_VAL);

	execute<slope_field>(m, v1d(m));
	execute<slope_field>(m, v2d(m));

#ifdef _RUN_MULTI
	linalg::vec::mesh vec1(linalg::variable<diffusion_var::v1>, m, v1d(m));
	linalg::vec::mesh vec2(linalg::variable<diffusion_var::v2>, m, v2d(m));
	linalg::vec::multi X(vec1, vec2);
	linalg::vec::mesh rhs1(linalg::variable<diffusion_var::v1>, m, rhs1d(m));
	linalg::vec::mesh rhs2(linalg::variable<diffusion_var::v2>, m, rhs2d(m));
	linalg::vec::multi RHS(rhs1, rhs2);

	auto A = linalg::discrete_operators::op_expr(
		make_boundary_operator_neumann(vec1),
		make_volume_operator(vec1),
		make_boundary_operator_neumann(vec2),
		make_volume_operator(vec2));
#else

	linalg::vec::mesh X(linalg::variable<diffusion_var::v1>, m, v1d(m));

	linalg::vec::mesh RHS(linalg::variable<diffusion_var::v1>, m, rhs1d(m));

	auto A = linalg::discrete_operators::op_expr(
		make_boundary_operator_neumann(X), make_volume_operator(X));
#endif

	RHS.set_scalar(0.0);

	linalg::krylov_params params(linalg::cg::settings{100, 1e-9, 1e-9},
	                             linalg::cg::topo_work<>::get(RHS),
	                             std::move(A));

	auto slv = linalg::op::create(std::move(params));

	auto info = slv.apply(RHS, X);

	flog(info) << "norm = " << info.res_norm_final << "\n";
	flog(info) << "iters = " << info.iters << "\n";

	return 0;
}
} // namespace diffusion