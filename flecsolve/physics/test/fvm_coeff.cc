#include <array>

#include <flecsi/exec/backend.hh>
#include <flecsi/flog.hh>
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/mesh.hh"
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
constexpr auto make_coeff_setters(const Vec &) {
	using namespace flecsolve::physics;

	auto constant_coeff = coefficient<constant_coefficient<Vec>, Vec>::create(
		{{3.14}, make_faces_ref(const_fd)});

	auto avg_coeff = coefficient<average_coefficient<Vec>, Vec>::create(
		{{1.0}, make_faces_ref(avg_fd)});

	return std::make_tuple(constant_coeff, avg_coeff);
}

template<class Vec>
constexpr auto make_bcs(const Vec &) {
	using namespace flecsolve::physics;

	return std::make_tuple(
		bc<dirichlet<Vec>, msh::x_axis, msh::boundary_low>::create({{1.0}}),
		bc<dirichlet<Vec>, msh::x_axis, msh::boundary_high>::create({{1.0}}),
		bc<dirichlet<Vec>, msh::y_axis, msh::boundary_low>::create({{1.0}}),
		bc<dirichlet<Vec>, msh::y_axis, msh::boundary_high>::create({{1.0}}),
		bc<dirichlet<Vec>, msh::z_axis, msh::boundary_low>::create({{1.0}}),
		bc<dirichlet<Vec>, msh::z_axis, msh::boundary_high>::create({{1.0}}));
}

static fvm_check chk_constx([](auto...) { return 3.14; },
                            "face coefficient constant [x]",
                            fconstant<msh::x_axis>{},
                            fconstant<msh::faces>{});
static fvm_check chk_consty([](auto...) { return 3.14; },
                            "face coefficient constant [y]",
                            fconstant<msh::y_axis>{},
                            fconstant<msh::faces>{});

static fvm_check chk_constz([](auto...) { return 3.14; },
                            "face coefficient constant [z]",
                            fconstant<msh::z_axis>{},
                            fconstant<msh::faces>{});

static fvm_check chk_avgx([](auto...) { return 1.0; },
                          "face coefficient as directional avg of cells [x]",
                          fconstant<msh::x_axis>{},
                          fconstant<msh::faces>{});

static fvm_check chk_avgy([](auto...) { return 1.0; },
                          "face coefficient as directional avg of cells [y]",
                          fconstant<msh::y_axis>{},
                          fconstant<msh::faces>{});

static fvm_check chk_avgz([](auto...) { return 1.0; },
                          "face coefficient as directional avg of cells [z]",
                          fconstant<msh::z_axis>{},
                          fconstant<msh::faces>{});

int fvm_coeff_test() {

	init_mesh({8, 8, 8});
	vec::mesh x(m, xd(m));

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

		auto [coef_const, coef_avg] = make_coeff_setters(x);

		coef_const.apply(x, x);
		coef_avg.apply(x, x);

		EXPECT_EQ((test<chk_constx>(m, const_fd[msh::x_axis](m))), 0);
		EXPECT_EQ((test<chk_consty>(m, const_fd[msh::y_axis](m))), 0);
		EXPECT_EQ((test<chk_constz>(m, const_fd[msh::z_axis](m))), 0);
		EXPECT_EQ((test<chk_avgx>(m, avg_fd[msh::x_axis](m))), 0);
		EXPECT_EQ((test<chk_avgy>(m, avg_fd[msh::y_axis](m))), 0);
		EXPECT_EQ((test<chk_avgz>(m, avg_fd[msh::z_axis](m))), 0);
	};
}

util::unit::driver<fvm_coeff_test> driver;

} // namespace physics_testing
} // namespace flecsolve
