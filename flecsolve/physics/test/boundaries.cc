#include <array>

#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/physics/boundary/dirichlet.hh"
#include "flecsolve/physics/boundary/neumann.hh"

#include "flecsolve/physics/specializations/operator_mesh.hh"

#include "test_setup.hh"

using namespace flecsi;

namespace flecsolve {
namespace physics_testing {

constexpr std::size_t NX = 16;
constexpr std::size_t NY = 16;

const field<double>::definition<msh, msh::cells> xd;
flecsi::util::key_array<field<double>::definition<msh, msh::faces>, msh::axes>
	bd;

template<class F, class T, class A, class D>
struct check {
	static constexpr double ftol = 1e-8;
	static constexpr auto OD =
		(A::value == msh::y_axis ? D::value : msh::logical);
	static constexpr auto ID =
		(A::value == msh::x_axis ? D::value : msh::logical);

	int operator()(msh::accessor<ro, ro> m, field<double>::accessor<ro, na> x) {

		UNIT (name) {

			auto xv = m.mdspan<msh::cells>(x);
			for (auto j : m.range<msh::cells, msh::y_axis, OD>()) {
				for (auto i : m.range<msh::cells, msh::x_axis, ID>()) {
					EXPECT_LT(std::abs(f(j, i) - xv[1][j][i]), ftol);
				}
			}
		};
	}

	F f;
	T name;
	A axconst;
	D dmconst;
};

template<class F, class T, class A, class D>
check(F &&, T &&, A &&, D &&) -> check<F, T, A, D>;

static check xlo{[](std::size_t, std::size_t) { return -1.0; },
                 "x low, dirichlet",
                 std::integral_constant<msh::axis, msh::x_axis>{},
                 std::integral_constant<msh::domain, msh::boundary_low>{}};

static check xhi{[](std::size_t, std::size_t) { return 1.0; },
                 "x high, dirichlet",
                 std::integral_constant<msh::axis, msh::x_axis>{},
                 std::integral_constant<msh::domain, msh::boundary_high>{}};

static check ylo{[](std::size_t, std::size_t) { return 1.0; },
                 "y low, neumann",
                 std::integral_constant<msh::axis, msh::y_axis>{},
                 std::integral_constant<msh::domain, msh::boundary_low>{}};
static check yhi{[](std::size_t, std::size_t) { return 1.0; },
                 "y hi, neumann",
                 std::integral_constant<msh::axis, msh::y_axis>{},
                 std::integral_constant<msh::domain, msh::boundary_high>{}};

template<class Vec>
constexpr auto make_bcs(const Vec &) {
	using namespace flecsolve::physics;
	auto bndry_xlo =
	 bc<dirichlet<Vec>, msh::x_axis, msh::boundary_low>::create({-1.0});


	auto bndry_xhi =
	bc<dirichlet<Vec>, msh::x_axis, msh::boundary_high>::create({1.0});

	auto bndry_ylo =
		bc<neumann<Vec>, msh::y_axis, msh::boundary_low>::create({});
	auto bndry_yhi =
		bc<neumann<Vec>, msh::y_axis, msh::boundary_high>::create({});

	return std::make_tuple(bndry_xlo, bndry_xhi, bndry_ylo, bndry_yhi);
}

int boundary_test() {

	init_mesh({NX, NY, 1});
	execute<fill_field<msh::faces>>(m, bd[msh::x_axis](m), 1.0);
	execute<fill_field<msh::faces>>(m, bd[msh::y_axis](m), 1.0);
	execute<fill_field<msh::faces>>(m, bd[msh::z_axis](m), 1.0);
	UNIT () {
		vec::mesh x(m, xd(m));

		auto [bndry_xlo, bndry_xhi, bndry_ylo, bndry_yhi] = make_bcs(x);
		x.set_scalar(1.0);

		bndry_xlo.apply(x, x);
		bndry_xhi.apply(x, x);
		bndry_ylo.apply(x, x);
		bndry_yhi.apply(x, x);

		EXPECT_EQ((test<xlo>(m, xd(m))), 0);
		EXPECT_EQ((test<xhi>(m, xd(m))), 0);
		EXPECT_EQ((test<ylo>(m, xd(m))), 0);
		EXPECT_EQ((test<yhi>(m, xd(m))), 0);
	};
}

unit::driver<boundary_test> driver;

} // namespace physics_testing
} // namespace flecsolve
