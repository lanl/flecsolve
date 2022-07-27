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
//#include "flecsolve/time-integrators/rk45.hh"
#include "flecsolve/util/config.hh"

#include "parameters.hh"
#include "state.hh"

using namespace flecsi;
namespace heat_eqn {
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

void init_field_ball(msh::accessor<ro, ro> vm,
                     field<scalar_t>::accessor<wo, na> xa) {
	auto xv = vm.mdspan<msh::cells>(xa);

	for (auto k : vm.range<msh::cells, msh::z_axis, msh::extended>()) {
		for (auto j : vm.range<msh::cells, msh::y_axis, msh::extended>()) {
			const auto y = (vm.value<msh::y_axis>(j) - 0.5);
			for (auto i : vm.range<msh::cells, msh::x_axis, msh::extended>()) {
				const auto x = (vm.value<msh::y_axis>(i) - 0.5);
				const auto r = std::sqrt(x * x + y * y);
				xv[k][j][i] = r > 0.3
				                  ? 1.0E-9
				                  : (0.3 - r); // vm.value<msh::x_axis>(i + 1);
			}
		}
	}
}

template<class Vec>
constexpr decltype(auto) make_boundary_operator(const Vec &) {
	using namespace flecsolve::physics;

	auto bndxl = bc<neumann<Vec>, msh::x_axis, msh::boundary_low>::create({});
	auto bndxh = bc<neumann<Vec>, msh::x_axis, msh::boundary_high>::create({});
	auto bndyl = bc<neumann<Vec>, msh::y_axis, msh::boundary_low>::create({});
	auto bndyh = bc<neumann<Vec>, msh::y_axis, msh::boundary_high>::create({});

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

template<class Vec>
decltype(auto) make_volume_operator(const Vec & v) {
	using namespace flecsolve::physics;

	flecsi::util::key_array<flecsi::field<scalar_t>::Reference<msh, msh::faces>,
	                        msh::axes>
		bref{diffb[msh::x_axis](m),
	         diffb[msh::y_axis](m),
	         diffb[msh::z_axis](m)};

	auto coeffop = unit_coefficent<Vec>::create({bref});
	auto voldiff = diffusion<Vec>::create({diffa(m), bref, 1.0, 0.0}, m);
	return op_expr(flecsolve::multivariable<Vec::var.value>, coeffop, voldiff);
}

int driver() {

	// initialize the mesh
	init_mesh();

	// fill auxiliary data fields
	execute<fill_field<msh::cells>>(m, diffa(m), DEFAULT_VAL);

	// set up initial conditions
	execute<init_field_ball>(m, ud(m));

	if (processes() < 100) {
		execute<check_vals>(m, ud(m), "u0");
	}
	else {
		flog(info) << "to see asci representation of ivs & solutions, run on a "
					  "single core.\n";
	}

	flecsolve::vec::mesh x(flecsolve::variable<heateqn_var::v1>, m, ud(m)), b(flecsolve::variable<heateqn_var::v1>,m,und(m));

	auto bnd_op =
		flecsolve::physics::op_expr(flecsolve::multivariable<heateqn_var::v1>,
	                                make_boundary_operator(x),
	                                make_boundary_operator_pseudo(x));

	// build the operator on the variables
	auto A =
		flecsolve::physics::op_expr(flecsolve::multivariable<heateqn_var::v1>,
	                                bnd_op,
	                                make_volume_operator(x));

	flecsolve::op::krylov_parameters params(flecsolve::cg::settings("solver"),
	                                        flecsolve::cg::topo_work<>::get(b),
	                                        std::ref(A));

	read_config("diffusion.cfg", params);

	b.set_scalar(0.0);
	// create the solver
	flecsolve::op::krylov slv(std::move(params));

	// run the solver
	auto info = slv.apply(b, x);
	// flecsolve::time_integrator::rk45::parameters params45(
	// 	"time-int",
	// 	std::ref(F),
	// 	flecsolve::time_integrator::rk45::topo_work<>::get(u));

	// read_config("heateqn.cfg", params45);
	// flecsolve::time_integrator::rk45::integrator ti45(std::move(params45));

	//  while(ti45.get_current_time() < ti45.get_final_time())
	//  {
	//  	ti45.advance(ti45.get_current_dt(), u, u);
	//  	ti45.update();
	// 	execute<check_vals>(m, ud(m), "u");
	//  }

	// // helper print of final solutions
	if (processes() < 100) {
		execute<check_vals>(m, ud(m), "u");
	}
	else {
		flog(info) << "to see asci representation of ivs & solutions, run on a "
					  "single core.\n";
	}

	return 0;
}
} // namespace diffusion
