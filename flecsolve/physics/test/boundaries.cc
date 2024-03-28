#include <array>

#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/physics/boundary/dirichlet.hh"
#include "flecsolve/physics/boundary/neumann.hh"

#include "test_setup.hh"

using namespace flecsi;

namespace flecsolve {
namespace physics_testing {

constexpr std::size_t NX = 16;
constexpr std::size_t NY = 16;

fld<msh::cells> xd;
flecsi::util::key_array<fld<msh::faces>, msh::axes> bd{};

auto xlo =
	std::tuple([](std::size_t, std::size_t, std::size_t) { return -1.0; },
               "x low, dirichlet");
auto xhi = std::tuple([](std::size_t, std::size_t, std::size_t) { return 1.0; },
                      "x high, dirichlet");
auto ylo = std::tuple([](std::size_t, std::size_t, std::size_t) { return 1.0; },
                      "y low, neumann");
auto yhi = std::tuple([](std::size_t, std::size_t, std::size_t) { return 1.0; },
                      "y high, neumann");

template<class Vec>
constexpr auto make_bcs(const Vec &) {
	using namespace flecsolve::physics;
	auto bndry_xlo =
		bc<dirichlet<Vec>, msh::x_axis, msh::domain::boundary_low>::create(
			{{-1.0}});

	auto bndry_xhi =
		bc<dirichlet<Vec>, msh::x_axis, msh::domain::boundary_high>::create(
			{{1.0}});

	auto bndry_ylo =
		bc<neumann<Vec>, msh::y_axis, msh::domain::boundary_low>::create({});
	auto bndry_yhi =
		bc<neumann<Vec>, msh::y_axis, msh::domain::boundary_high>::create({});

	return std::make_tuple(bndry_xlo, bndry_xhi, bndry_ylo, bndry_yhi);
}

int boundary_test() {
	msh::slot m;
	msh::cslot coloring;

	init_mesh(m, coloring, {NX, NY, 1});
	execute<fill_field<msh::faces>>(m, bd[msh::x_axis](m), 1.0);
	execute<fill_field<msh::faces>>(m, bd[msh::y_axis](m), 1.0);
	execute<fill_field<msh::faces>>(m, bd[msh::z_axis](m), 1.0);
	UNIT () {
		auto x = vec::make(m, xd(m));

		auto [bndry_xlo, bndry_xhi, bndry_ylo, bndry_yhi] = make_bcs(x);
		x.set_scalar(1.0);

		bndry_xlo.apply(x, x);
		bndry_xhi.apply(x, x);
		bndry_ylo.apply(x, x);
		bndry_yhi.apply(x, x);

		// execute<check_vals>(m, xd(m), "checks");
		EXPECT_TRUE(fvm_run<r_lo<msh::x_axis>>(xlo, m, xd(m)));
		EXPECT_TRUE(fvm_run<r_hi<msh::x_axis>>(xhi, m, xd(m)));
		EXPECT_TRUE(fvm_run<r_lo<msh::y_axis>>(ylo, m, xd(m)));
		EXPECT_TRUE(fvm_run<r_hi<msh::y_axis>>(yhi, m, xd(m)));
	};
	return 0;
}

util::unit::driver<boundary_test> driver;

} // namespace physics_testing
} // namespace flecsolve
