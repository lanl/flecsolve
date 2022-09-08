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
#include "flecsolve/solvers/solver_settings.hh"

using namespace flecsi;

namespace eqdiff {

//===================================================
//=============== problem setup =====================
//===================================================
using scalar_t = double;

// define the problem dimensions
constexpr std::size_t NX = 8;
constexpr std::size_t NY = 8;
constexpr std::size_t NZ = 1;

// declare the "variable"'s of the "multivector"
constexpr std::size_t NVAR = 2;
enum class diffusion_var { v1 = 0, v2 };

// a useful number
constexpr scalar_t DEFAULT_VAL = 1.0;

//===================================================
//=============== flecsi machinary ==================
//===================================================
using msh = flecsolve::physics::fvm_narray;

msh::slot m;
msh::cslot coloring;

// field definition, & reference
template<auto S>
using fld = const field<scalar_t>::definition<msh, S>;

template<auto S>
using fldr = field<scalar_t>::Reference<msh, S>;

// a "vector" field with `x`,`y`,`z` components
template<auto S>
using vec_fld = util::key_array<fld<S>, msh::axes>;

template<auto S>
using vec_fldr = util::key_array<fldr<S>, msh::axes>;

// useful to produce an array of field references from an array of field defs
const inline auto make_faces_ref(const vec_fld<msh::faces> & fs) {
	return vec_fldr<msh::faces>{
		fs[msh::x_axis](m), fs[msh::y_axis](m), fs[msh::z_axis](m)};
}

//===================================================
//=============== field storage =====================
//===================================================

// `xd`'s are the solution fields
// `rhsd`s are the right-hand side fields
// `diff_srcd`'s are a field of source terms
std::array<fld<msh::cells>, NVAR> xd{}, rhsd{}, diff_srcd{};
// `diff_coeffd`'s are the `x`,`y`,`z` fields of face-centered coefficient
// values
std::array<vec_fld<msh::faces>, NVAR> diff_coeffd{};

//===================================================
//=============== utility functions==================
//===================================================

void init_mesh() {
	std::vector<std::size_t> extents{{NX, NY, NZ}};
	auto colors = msh::distribute(processes(), extents);
	coloring.allocate(colors, extents);

	msh::gbox geometry;
	geometry[msh::x_axis][0] = 0.0;
	geometry[msh::x_axis][1] = 1.0;
	geometry[msh::y_axis] = geometry[msh::x_axis];
	geometry[msh::z_axis] = geometry[msh::x_axis];

	m.allocate(coloring.get(), geometry);
	run::context::instance().add_topology(m);
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

/**
 * constructs full-boundary neumann conditions
 */
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

/**
 * constructs full-boundary dirichlet conditions
 */
template<class Vec>
constexpr decltype(auto) make_boundary_operator_dirichlet(const Vec &) {
	using namespace flecsolve::physics;

	auto bndxl =
		bc<dirichlet<Vec>, msh::x_axis, msh::boundary_low>::create({1.0E-9});
	auto bndxh =
		bc<dirichlet<Vec>, msh::x_axis, msh::boundary_high>::create({1.0E-9});
	auto bndyl =
		bc<dirichlet<Vec>, msh::y_axis, msh::boundary_low>::create({1.0E-9});
	auto bndyh =
		bc<dirichlet<Vec>, msh::y_axis, msh::boundary_high>::create({1.0E-9});
	return op_expr(
		flecsolve::multivariable<Vec::var.value>, bndxl, bndxh, bndyl, bndyh);
}

/**
 * used to make occupy the "unused" `z-axis` boundary for the 2D problem
 */
template<class Vec>
constexpr decltype(auto) make_boundary_operator_pseudo(const Vec &) {
	using namespace flecsolve::physics;

	auto bndl = bc<neumann<Vec>, msh::z_axis, msh::boundary_low>::create({});
	auto bndh = bc<neumann<Vec>, msh::z_axis, msh::boundary_high>::create({});
	return op_expr(flecsolve::multivariable<Vec::var.value>, bndl, bndh);
}

/**
 * constructs the diffusion operator.
 *
 * the coefficent object is used to set the face-centered coefficent values
 * used by the diffusion operator
 */
template<std::size_t N, class Vec>
decltype(auto)
make_volume_operator(const Vec & v, scalar_t beta, scalar_t alpha) {
	using namespace flecsolve::physics;

	auto constant_coeff = coefficient<constant_coefficient<Vec>, Vec>::create(
		{1.0, make_faces_ref(diff_coeffd[N])});
	auto voldiff = diffusion<Vec>::create(
		{diff_srcd[N](m), make_faces_ref(diff_coeffd[N]), beta, alpha}, m);
	return op_expr(
		flecsolve::multivariable<Vec::var.value>, constant_coeff, voldiff);
}

/**
 * utility routine to construct the multivectors from arrays of fields
 */
namespace detail {
template<class FieldDefArr, std::size_t... I>
decltype(auto) make_multivector(const FieldDefArr & fd,
                                std::index_sequence<I...>) {
	using namespace flecsolve;
	return vec::multi{
		vec::mesh(variable<static_cast<diffusion_var>(I)>, m, fd[I](m))...};
}
}

template<class FieldDefArr>
decltype(auto) make_multivector(const FieldDefArr & fd) {
	return detail::make_multivector(fd, std::make_index_sequence<NVAR>{});
}

inline int driver() {

	// initialize the mesh
	init_mesh();

	//===================================================
	//=============== multivectors ======================
	//===================================================

	auto X = make_multivector(xd);
	auto RHS = make_multivector(rhsd);

	auto & [vec1, vec2] = X;

	// set both to scalar
	X.set_scalar(2.0);

	// set the RHS to vanish
	RHS.set_scalar(0.0);

	//===================================================
	//=============== operators =========================
	//===================================================

	// diffusion operator parameters
	std::array<scalar_t, NVAR> diff_param_alpha, diff_param_beta;

	for (unsigned i = 0; i < NVAR; ++i) {
		diff_param_alpha[i] = 0.0;
		diff_param_beta[i] = 1.0;
	}

	// construct the dirichlet boundary for the `v1` vector
	// the `z-axis` is treated as zero-flux
	auto bnd_op_1 =
		flecsolve::physics::op_expr(flecsolve::multivariable<diffusion_var::v1>,
	                                make_boundary_operator_dirichlet(vec1),
	                                make_boundary_operator_pseudo(vec1));

	// neumann boundary for the `v2` vector
	auto bnd_op_2 =
		flecsolve::physics::op_expr(flecsolve::multivariable<diffusion_var::v2>,
	                                make_boundary_operator_neumann(vec2),
	                                make_boundary_operator_pseudo(vec2));

	// build the full operator on the variables
	// notice we can use both operator objects and previous expressions
	auto A = flecsolve::physics::op_expr(
		flecsolve::multivariable<diffusion_var::v1, diffusion_var::v2>,
		bnd_op_1,
		make_volume_operator<0>(vec1, diff_param_beta[0], diff_param_alpha[0]),
		bnd_op_2,
		make_volume_operator<1>(vec2, diff_param_beta[1], diff_param_alpha[1]));

	//===================================================
	//=============== solver ============================
	//===================================================

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
		execute<check_vals>(m, xd[0](m), "v1");
		//"[variable1] solution with zero-flux boundary (n ⋅ ∇ u=0 on ∂Ω) ");
		execute<check_vals>(m, xd[1](m), "v2");
		//"[variable2] solution vanishes at boundary (u=0 on ∂Ω)");
	}
	else {
		flog(info) << "to see asci representation of ivs & solutions, run on a "
					  "single core.\n";
	}

	return 0;
}
} // namespace
