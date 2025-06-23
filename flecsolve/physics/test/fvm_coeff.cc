#include <array>

#include <flecsi/exec/backend.hh>
#include <flecsi/flog.hh>
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/physics/specializations/fvm_narray.hh"
#include "flecsolve/physics/volume_diffusion/coefficient.hh"
#include "flecsolve/physics/boundary/dirichlet.hh"

#include "test_setup.hh"

using namespace flecsi;
namespace flecsolve {
namespace physics_testing {

fld<msh::cells> xd;
util::key_array<fld<msh::faces>, msh::axes> const_fd{}, avg_fd{};

template<class Vec>
constexpr auto make_coeff_setters(msh::topology & m, const Vec &) {
	using namespace flecsolve::physics;

	auto constant_coeff = coefficient<constant_coefficient<Vec>, Vec>::create(
		{{3.14}, make_faces_ref(m, const_fd)});

	auto avg_coeff = coefficient<average_coefficient<Vec>, Vec>::create(
		{{1.0}, make_faces_ref(m, avg_fd)});

	return std::make_tuple(constant_coeff, avg_coeff);
}

template<class Vec>
constexpr auto make_bcs(const Vec &) {
	using namespace flecsolve::physics;

	return std::make_tuple(
		bc<dirichlet<Vec>, msh::x_axis, msh::domain::boundary_low>::create(
			{{1.0}}),
		bc<dirichlet<Vec>, msh::x_axis, msh::domain::boundary_high>::create(
			{{1.0}}),
		bc<dirichlet<Vec>, msh::y_axis, msh::domain::boundary_low>::create(
			{{1.0}}),
		bc<dirichlet<Vec>, msh::y_axis, msh::domain::boundary_high>::create(
			{{1.0}}),
		bc<dirichlet<Vec>, msh::z_axis, msh::domain::boundary_low>::create(
			{{1.0}}),
		bc<dirichlet<Vec>, msh::z_axis, msh::domain::boundary_high>::create(
			{{1.0}}));
}

// test setup for constant coefficient values
auto chk_constx = std::make_tuple([](auto...) { return 3.14; },
                                  "face coefficient constant [x]");
auto chk_consty = std::make_tuple([](auto...) { return 3.14; },
                                  "face coefficient constant [y]");
auto chk_constz = std::make_tuple([](auto...) { return 3.14; },
                                  "face coefficient constant [z]");

// test setup for average coefficient values through a direction
auto chk_avgx =
	std::make_tuple([](auto...) { return 1.0; },
                    "face coefficient as directional avg of cells [x]");
auto chk_avgy =
	std::make_tuple([](auto...) { return 1.0; },
                    "face coefficient as directional avg of cells [y]");
auto chk_avgz =
	std::make_tuple([](auto...) { return 1.0; },
                    "face coefficient as directional avg of cells [z]");

int fvm_coeff_test(flecsi::scheduler & s) {
	msh::ptr mptr;

	auto & m = init_mesh(s, mptr, {8, 8, 8});
	auto x = vec::make(xd(m));

	UNIT () {
		x.set_scalar(1.0);

		// set boundaries (manual)
		auto [bxl, bxh, byl, byh, bzl, bzh] = make_bcs(x);
		bxl.apply(x, x);
		bxh.apply(x, x);
		byl.apply(x, x);
		byh.apply(x, x);
		bzl.apply(x, x);
		bzh.apply(x, x);

		auto [coef_const, coef_avg] = make_coeff_setters(m, x);

		coef_const.apply(x, x);
		coef_avg.apply(x, x);

		EXPECT_TRUE(fvm_run<rface<msh::x_axis>>(
			chk_constx, m, const_fd[msh::x_axis](m)));

		EXPECT_TRUE(fvm_run<rface<msh::y_axis>>(
			chk_consty, m, const_fd[msh::y_axis](m)));
		EXPECT_TRUE(fvm_run<rface<msh::z_axis>>(
			chk_constz, m, const_fd[msh::z_axis](m)));

		EXPECT_TRUE(
			fvm_run<rface<msh::x_axis>>(chk_avgx, m, avg_fd[msh::x_axis](m)));
		EXPECT_TRUE(
			fvm_run<rface<msh::y_axis>>(chk_avgy, m, avg_fd[msh::y_axis](m)));
		EXPECT_TRUE(
			fvm_run<rface<msh::z_axis>>(chk_avgz, m, avg_fd[msh::z_axis](m)));
	};
}

util::unit::driver<fvm_coeff_test> driver;

} // namespace physics_testing
} // namespace flecsolve
