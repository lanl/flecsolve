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
#include <flecsi/util/constant.hh>

#include <iomanip>
#include <iostream>

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/util/config.hh"
#include "flecsolve/physics/boundary/dirichlet.hh"
#include "flecsolve/physics/boundary/neumann.hh"
#include "flecsolve/physics/expressions/operator_expression.hh"
#include "flecsolve/physics/volume_diffusion/diffusion.hh"
#include "flecsolve/physics/volume_diffusion/coefficient.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/solver_settings.hh"
#include "flecsolve/operators/core.hh"

using namespace flecsi;

namespace eqdiff {

//===================================================
//=============== problem setup =====================
//===================================================
using scalar_t = double;

// define the problem dimensions
inline flecsi::program_option<std::size_t>
	NX("NX", "The x extents of the mesh.", 1);
inline flecsi::program_option<std::size_t>
	NY("NY", "The y extents of the mesh.", 1);

// declare the "variable"'s of the "multivector"
constexpr std::size_t NVAR = 2;
enum class diffusion_var { v1 = 0, v2 };

// a useful number
constexpr scalar_t DEFAULT_VAL = 1.0;

//===================================================
//=============== flecsi machinary ==================
//===================================================
using msh = flecsolve::physics::fvm_narray;

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
inline auto make_faces_ref(msh::slot & m, const vec_fld<msh::faces> & fs) {
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

void init_mesh(msh::slot & m) {
	using mbase = msh::base;

	std::vector<flecsi::util::gid> extents{{NX.value(), NY.value(), 1}};
	msh::index_definition idef, idef_faces;
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

	m.allocate(msh::mpi_coloring(idef, idef_faces), geometry);

	run::context::instance().add_topology(m);
}

/**
 * constructs full-boundary neumann conditions
 */
template<class Vec>
constexpr decltype(auto) make_boundary_operator_neumann(const Vec &) {
	using namespace flecsolve::physics;

	auto bndxl =
		bc<neumann<Vec>, msh::x_axis, msh::domain::boundary_low>::create({});
	auto bndxh =
		bc<neumann<Vec>, msh::x_axis, msh::domain::boundary_high>::create({});
	auto bndyl =
		bc<neumann<Vec>, msh::y_axis, msh::domain::boundary_low>::create({});
	auto bndyh =
		bc<neumann<Vec>, msh::y_axis, msh::domain::boundary_high>::create({});

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
		bc<dirichlet<Vec>, msh::x_axis, msh::domain::boundary_low>::create(
			{1.0E-9});
	auto bndxh =
		bc<dirichlet<Vec>, msh::x_axis, msh::domain::boundary_high>::create(
			{1.0E-9});
	auto bndyl =
		bc<dirichlet<Vec>, msh::y_axis, msh::domain::boundary_low>::create(
			{1.0E-9});
	auto bndyh =
		bc<dirichlet<Vec>, msh::y_axis, msh::domain::boundary_high>::create(
			{1.0E-9});
	return op_expr(
		flecsolve::multivariable<Vec::var.value>, bndxl, bndxh, bndyl, bndyh);
}

/**
 * used to make occupy the "unused" `z-axis` boundary for the 2D problem
 */
template<class Vec>
constexpr decltype(auto) make_boundary_operator_pseudo(const Vec &) {
	using namespace flecsolve::physics;

	auto bndl =
		bc<neumann<Vec>, msh::z_axis, msh::domain::boundary_low>::create({});
	auto bndh =
		bc<neumann<Vec>, msh::z_axis, msh::domain::boundary_high>::create({});
	return op_expr(flecsolve::multivariable<Vec::var.value>, bndl, bndh);
}

/**
 * constructs the diffusion operator.
 *
 * the coefficent object is used to set the face-centered coefficent values
 * used by the diffusion operator
 */
template<std::size_t N, class Vec>
decltype(auto) make_volume_operator(msh::slot & m,
                                    const Vec &,
                                    scalar_t beta,
                                    scalar_t alpha) {
	using namespace flecsolve::physics;

	auto constant_coeff = coefficient<constant_coefficient<Vec>, Vec>::create(
		{{1.0}, make_faces_ref(m, diff_coeffd[N])});
	auto voldiff = diffusion<Vec>::create(
		{diff_srcd[N](m), make_faces_ref(m, diff_coeffd[N]), beta, alpha}, m);
	return op_expr(
		flecsolve::multivariable<Vec::var.value>, constant_coeff, voldiff);
}

/**
 * utility routine to construct the multivectors from arrays of fields
 */
namespace detail {
template<class FieldDefArr, std::size_t... I>
decltype(auto) make_multivector(msh::slot & m,
                                const FieldDefArr & fd,
                                std::index_sequence<I...>) {
	using namespace flecsolve;
	return vec::make(
		vec::make(variable<static_cast<diffusion_var>(I)>, m, fd[I](m))...);
}

template<std::size_t I>
void field_out(msh::accessor<ro, ro> vm,
               field<scalar_t>::accessor<wo, na> xa,
               std::ofstream & ofs) {
	auto xv = vm.mdspan<msh::cells>(xa);

	for (auto k : vm.range<msh::cells, msh::z_axis, msh::domain::all>()) {
		const scalar_t z = vm.value<msh::y_axis>(k);
		for (auto j : vm.range<msh::cells, msh::y_axis, msh::domain::all>()) {
			const scalar_t y = vm.value<msh::y_axis>(j);
			for (auto i :
			     vm.range<msh::cells, msh::x_axis, msh::domain::all>()) {
				const scalar_t x = vm.value<msh::x_axis>(i);
				ofs << I << ": " << i << " " << j << " " << k << " " << x << " "
					<< y << " " << z << " " << xv[k][j][i] << "\n";
			}
		}
	}
}

template<class FieldDefArr, std::size_t... I>
void fields_out(msh::slot & m,
                const FieldDefArr & fd,
                std::ofstream & of,
                std::index_sequence<I...>) {
	using namespace flecsolve;
	(flecsi::execute<field_out<I>, flecsi::mpi>(m, fd[I](m), of), ...);
}
} // namespace detail

template<class FieldDeffArr>
void fields_out(msh::slot & m, FieldDeffArr & fd, std::string filen) {

	std::stringstream ss;
	ss << filen << "_" << process() << ".dat";
	std::ofstream of(ss.str(), std::ofstream::out);

	detail::fields_out(m, fd, of, std::make_index_sequence<NVAR>{});
}

template<class FieldDefArr>
decltype(auto) make_multivector(msh::slot & m, const FieldDefArr & fd) {
	return detail::make_multivector(m, fd, std::make_index_sequence<NVAR>{});
}

inline int driver() {

	flog(info) << "multivector 2D diffusion: \n";
	flog(info) << "nranks = " << processes() << "\n";

	flog(info) << "initializing mesh\n";
	// initialize the mesh
	msh::slot m;

	init_mesh(m);

	//===================================================
	//=============== multivectors ======================
	//===================================================

	auto X = make_multivector(m, xd);
	auto RHS = make_multivector(m, rhsd);

	auto & [vec1, vec2] = X;

	flog(info) << "setting initial multivectors\n";
	// set both to scalar
	X.set_scalar(2.0);

	// set the RHS to vanish
	RHS.set_scalar(0.0);

	//===================================================
	//=============== operators =========================
	//===================================================

	flog(info) << "constructing operators\n";

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
	auto A = flecsolve::op::make(flecsolve::physics::op_expr(
		flecsolve::multivariable<diffusion_var::v1, diffusion_var::v2>,
		bnd_op_1,
		make_volume_operator<0>(
			m, vec1, diff_param_beta[0], diff_param_alpha[0]),
		bnd_op_2,
		make_volume_operator<1>(
			m, vec2, diff_param_beta[1], diff_param_alpha[1])));

	//===================================================
	//=============== solver ============================
	//===================================================

	flog(info) << "constructing solver\n";
	// create the solver
	auto slv = flecsolve::cg::solver(
		flecsolve::read_config("diffusion.cfg", flecsolve::cg::options("solver")),
		flecsolve::cg::make_work(RHS))(flecsolve::op::ref(A));

	flog(info) << "applying the solver\n";
	// run the solver
	auto info = slv.apply(RHS, X);

	flog(info) << "apply complete\n";
	// print some statistics on the solve
	flog(info) << "norm = " << info.res_norm_final << "\n";
	flog(info) << "iters = " << info.iters << "\n";

	// helper print of final solutions
	std::string file_final = "ed_final";

	flog(info) << "writing out solution to " << file_final << "_N.dat\n";

	fields_out(m, xd, file_final);

	return 0;
}
} // namespace
