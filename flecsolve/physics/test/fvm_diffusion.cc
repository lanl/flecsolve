#include <array>

#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/physics/boundary/dirichlet.hh"
#include "flecsolve/physics/boundary/neumann.hh"
#include "flecsolve/physics/expressions/operator_expression.hh"
#include "flecsolve/physics/volume_diffusion/diffusion.hh"
#include "flecsolve/physics/volume_diffusion/coefficient.hh"

#include "test_setup.hh"

using namespace flecsi;

namespace flecsolve {
namespace physics_testing {

fld<msh::cells> xd, yd;

fld<msh::cells> ad;
flecsi::util::key_array<fld<msh::faces>, msh::axes> bd{};

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
	auto bndzl =
		bc<neumann<Vec>, msh::z_axis, msh::domain::boundary_low>::create({});
	auto bndzh =
		bc<neumann<Vec>, msh::z_axis, msh::domain::boundary_high>::create({});

	return op_expr(flecsolve::multivariable<Vec::var.value>,
	               bndxl,
	               bndxh,
	               bndyl,
	               bndyh,
	               bndzl,
	               bndzh);
}

template<class Vec>
constexpr decltype(auto) make_boundary_operator_dirichlet(const Vec &) {
	using namespace flecsolve::physics;

	auto bndxl =
		bc<dirichlet<Vec>, msh::x_axis, msh::domain::boundary_low>::create(
			{{1.0E-9}});
	auto bndxh =
		bc<dirichlet<Vec>, msh::x_axis, msh::domain::boundary_high>::create(
			{{1.0E-9}});
	auto bndyl =
		bc<dirichlet<Vec>, msh::y_axis, msh::domain::boundary_low>::create(
			{{1.0E-9}});
	auto bndyh =
		bc<dirichlet<Vec>, msh::y_axis, msh::domain::boundary_high>::create(
			{{1.0E-9}});
	auto bndzl =
		bc<dirichlet<Vec>, msh::z_axis, msh::domain::boundary_low>::create(
			{{1.0E-9}});
	auto bndzh =
		bc<dirichlet<Vec>, msh::z_axis, msh::domain::boundary_high>::create(
			{{1.0E-9}});

	return op_expr(flecsolve::multivariable<Vec::var.value>,
	               bndxl,
	               bndxh,
	               bndyl,
	               bndyh,
	               bndzl,
	               bndzh);
}

template<class Vec>
decltype(auto)
make_volume_operator(const Vec &, scalar_t beta, scalar_t alpha) {
	using namespace flecsolve::physics;

	// auto vd = operator_creator<diffusion<Vec>,
	// constant_coefficent>::create(bref, diffa[N](m), 1.0, 0.0, m);
	auto constant_coeff = coefficient<constant_coefficient<Vec>, Vec>::create(
		{{1.0}, make_faces_ref(bd)});

	auto voldiff =
		diffusion<Vec>::create({ad(m), make_faces_ref(bd), beta, alpha}, m);
	return op_expr(
		flecsolve::multivariable<Vec::var.value>, constant_coeff, voldiff);
}

static fvm_check zero_flux{[](auto...) { return 0.0; }, "flux is zero"};

static inline int source_only(msh::accessor<ro, ro> m,
                              field<scalar_t>::accessor<ro, na> xa,
                              scalar_t alpha) {

	auto xv = m.mdspan<msh::cells>(xa);
	auto kk = m.range<msh::cells, msh::z_axis, msh::domain::logical>();
	auto jj = m.range<msh::cells, msh::y_axis, msh::domain::logical>();
	auto ii = m.range<msh::cells, msh::x_axis, msh::domain::logical>();
	auto vol = m.volume();

	UNIT ("source only") {
		for (auto k : kk) {
			for (auto j : jj) {

				for (auto i : ii) {

					EXPECT_LT(std::abs(xv[k][j][i] - alpha * vol), DEFAULT_TOL);
				}
			}
		}
	};
}

static inline int boundary_sink(msh::accessor<ro, ro> m,
                                field<scalar_t>::accessor<ro, na> xa,
                                scalar_t x0) {
	auto xv = m.mdspan<msh::cells>(xa);
	auto kk = m.range<msh::cells, msh::z_axis, msh::domain::logical>();
	auto jj = m.range<msh::cells, msh::y_axis, msh::domain::logical>();
	auto ii = m.range<msh::cells, msh::x_axis, msh::domain::logical>();
	std::ostringstream oss;

	oss << "====================\n";

	oss << "--------------------\n";
	UNIT ("boundry sink") {
		for (auto k : kk) {
			oss << "-- k = " << k << "---------\n";
			for (auto j : jj) {
				oss << "j = " << j << std::setw(6) << " | ";
				for (auto i : ii) {
					if ((i == ii.front() || i == ii.back()) ||
					    (j == jj.front() || j == jj.back()) ||
					    (k == kk.front() || k == kk.back())) {
						EXPECT_LT(xv[k][j][i], x0);
					}
					else {
						EXPECT_LT(std::abs(xv[k][j][i]), DEFAULT_TOL);
					}
					oss << xv[k][j][i] << " ";
				}
				oss << "\n";
			}
			oss << "--------------------\n";
		}

		oss << "====================\n";

		oss << "\n";
		std ::cout << oss.str();
	};
}

int fvm_diffusion_test() {

	init_mesh({8, 8, 8});
	vec::mesh x(m, xd(m));
	vec::mesh y(m, yd(m));
	vec::mesh a(m, ad(m));

	a.set_scalar(1.0);

	auto bc_dir = make_boundary_operator_dirichlet(x);
	auto bc_neu = make_boundary_operator_neumann(x);

	UNIT () {
		{
			auto beta = 1.0;
			auto vol_op = make_volume_operator(x, beta, 0.0);
			auto diff_op = flecsolve::physics::op_expr(
				flecsolve::multivariable<decltype(x)::var.value>,
				bc_neu,
				vol_op);
			x.set_scalar(1.0);
			diff_op.apply(x, y);

			EXPECT_EQ((test<zero_flux>(m, yd(m))), 0);
		}
		{
			auto alpha = 1.0;
			auto vol_op = make_volume_operator(x, 0.0, alpha);
			auto diff_op = flecsolve::physics::op_expr(
				flecsolve::multivariable<decltype(x)::var.value>,
				bc_neu,
				vol_op);
			diff_op.apply(x, y);
			EXPECT_EQ((test<source_only>(m, yd(m), alpha)), 0);
		}
		{
			auto fc = 1.0;
			auto vol_op = make_volume_operator(x, 1.0, 0.0);
			auto diff_op = flecsolve::physics::op_expr(
				flecsolve::multivariable<decltype(x)::var.value>,
				bc_dir,
				vol_op);
			x.set_scalar(fc);
			y.set_scalar(0.0);
			diff_op.apply(x, y);
			EXPECT_EQ((test<boundary_sink>(m, yd(m), fc)), 0);
		}
	};
}

util::unit::driver<fvm_diffusion_test> driver;

} // namespace physics_testing
} // namespace flecsolve
