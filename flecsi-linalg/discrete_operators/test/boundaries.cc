#include <array>

#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/vectors/multi.hh"

#include "flecsi-linalg/discrete_operators/boundary/dirichlet.hh"
#include "flecsi-linalg/discrete_operators/boundary/neumann.hh"

#include "flecsi-linalg/discrete_operators/specializations/operator_mesh.hh"

using namespace flecsi;

namespace flecsi::linalg {
using msh = discrete_operators::operator_mesh;

constexpr std::size_t NX = 16;
constexpr std::size_t NY = 16;

msh::slot m;
msh::cslot coloring;

const field<double>::definition<msh, msh::cells> xd;
const field<double>::definition<msh, msh::faces> bd;

enum class bndvar { v1 = 1, v2 };

void init_mesh() {
	std::vector<std::size_t> extents{{NX, NY}};
	auto colors = msh::distribute(flecsi::processes(), extents);
	coloring.allocate(colors, extents);

	msh::grect geometry;
	geometry[0][0] = 0.0;
	geometry[0][1] = 1.0;
	geometry[1] = geometry[0];

	m.allocate(coloring.get(), geometry);
}

// TODO: make common, possibly replace tasks::zero_op
template<auto Space>
void fill_field(msh::accessor<ro, ro> vm,
                field<double>::accessor<wo, na> xa,
                double val)

{
	auto xv = vm.mdspan<Space>(xa);
	for (auto j : vm.range<Space, msh::y_axis, msh::all>()) {
		for (auto i : vm.range<Space, msh::x_axis, msh::all>()) {
			xv[j][i] = val;
		}
	}
}

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
					EXPECT_LT(std::abs(f(j, i) - xv[j][i]), ftol);
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

int boundary_test() {

	init_mesh();
	execute<fill_field<msh::faces>>(m, bd(m), 1.0);
	UNIT () {
		vec::mesh x(linalg::variable<bndvar::v1>, m, xd(m));

		auto bndry_xlo = discrete_operators::make_operator<
			discrete_operators::
				dirichlet<bndvar::v1, msh, msh::x_axis, msh::boundary_low>>(
			-1.0);

		auto bndry_xhi = discrete_operators::make_operator<
			discrete_operators::
				dirichlet<bndvar::v1, msh, msh::x_axis, msh::boundary_high>>(
			1.0);

		auto bndry_ylo = discrete_operators::make_operator<
			discrete_operators::
				neumann<bndvar::v1, msh, msh::y_axis, msh::boundary_low>>(
			bd(m));
		auto bndry_yhi = discrete_operators::make_operator<
			discrete_operators::
				neumann<bndvar::v1, msh, msh::y_axis, msh::boundary_high>>(
			bd(m));

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

} // namespace flecsi::linalg
