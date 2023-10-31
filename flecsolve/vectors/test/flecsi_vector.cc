#include <complex>

#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsolve/vectors/mesh.hh"

#include "flecsolve/util/test/mesh.hh"

using namespace flecsi;

namespace flecsolve {

testmesh::slot msh;
testmesh::cslot coloring;

using realf = field<double>;
using compf = field<std::complex<double>>;

const realf::definition<testmesh, testmesh::cells> xd, yd, zd, tmpd;
const compf::definition<testmesh, testmesh::cells> xd_c, yd_c, zd_c, tmpd_c;

template<int index>
static constexpr double rconv(double gid) {
	return (index + 1) * gid;
}

template<int index>
static constexpr std::complex<double> cconv(double gid) {
	if constexpr (index == 0)
		return {.3 * gid, 0.7 * gid};
	else if constexpr (index == 1)
		return {.1 * gid, .8 * gid};
	else
		return {.5 * gid, .4 * gid};
}

template<class T>
static constexpr std::complex<double> conv(T c) {
	return c;
}

void init_fields(testmesh::accessor<ro, ro> m,
                 realf::accessor<wo, na> xa,
                 realf::accessor<wo, na> ya,
                 realf::accessor<wo, na> za,
                 compf::accessor<wo, na> xa_c,
                 compf::accessor<wo, na> ya_c,
                 compf::accessor<wo, na> za_c) {
	for (auto dof : m.dofs<testmesh::cells>()) {
		double gid = m.global_id(dof);
		xa[dof] = rconv<0>(gid);
		ya[dof] = rconv<1>(gid);
		za[dof] = rconv<2>(gid);

		xa_c[dof] = cconv<0>(gid);
		ya_c[dof] = cconv<1>(gid);
		za_c[dof] = cconv<2>(gid);
	}
}
static constexpr double ftol = 1e-8;

template<class FN>
decltype(auto) check_f(FN && fn,
                       testmesh::accessor<ro, ro> m,
                       realf::accessor<ro, na> x,
                       compf::accessor<ro, na> x_c) {
	UNIT (fn.second) {
		for (auto dof : m.dofs<testmesh::cells>()) {
			auto gid = m.global_id(dof);
			auto [rans, cans] = fn.first(gid);
			EXPECT_LT(std::abs(rans - x[dof]), ftol);
			EXPECT_LT(std::abs(cans - x_c[dof]), ftol);
		}
	};
}

using namespace std::complex_literals;
auto add = std::make_pair(
	[](double gid) {
		return std::make_pair(rconv<0>(gid) + rconv<2>(gid),
	                          cconv<0>(gid) + cconv<2>(gid));
	},
	"add");

auto subtract = std::make_pair(
	[](double gid) {
		return std::make_pair(rconv<0>(gid) - rconv<2>(gid),
	                          cconv<0>(gid) - cconv<2>(gid));
	},
	"subtract");
auto mult = std::make_pair(
	[](double gid) {
		return std::make_pair(rconv<0>(gid) * rconv<2>(gid),
	                          cconv<0>(gid) * cconv<2>(gid));
	},
	"multiply");
auto scalar_add = std::make_pair(
	[](double gid) {
		return std::make_pair(rconv<0>(gid) + 1,
	                          cconv<0>(gid) + conv((1. + 1i)));
	},
	"add scalar");
auto divide = std::make_pair(
	[](double gid) {
		return std::make_pair(rconv<1>(gid) / (rconv<0>(gid) + 1),
	                          cconv<1>(gid) /
	                              (cconv<0>(gid) + conv((1. + 1i))));
	},
	"divide");
auto scale = std::make_pair(
	[](double gid) {
		return std::make_pair(rconv<0>(gid) * 2, cconv<0>(gid) * 2.4);
	},
	"scale");
auto recip = std::make_pair(
	[](double gid) {
		return std::make_pair(1.0 / (rconv<1>(gid) + 1),
	                          1.0 / (cconv<1>(gid) + conv((1. + 1i))));
	},
	"reciprocal");
auto linsum = std::make_pair(
	[](double gid) {
		return std::make_pair(rconv<1>(gid) * 8 + rconv<2>(gid) * 9,
	                          cconv<1>(gid) * 8. + cconv<2>(gid) * 9.);
	},
	"linear sum");
auto axpy = std::make_pair(
	[](double gid) {
		return std::make_pair(rconv<0>(gid) * 7 + rconv<1>(gid),
	                          cconv<0>(gid) * conv((7. + 3i)) + cconv<1>(gid));
	},
	"axpy");
auto axpby = std::make_pair(
	[](double gid) {
		return std::make_pair(rconv<2>(gid) * 4 + rconv<1>(gid) * 11,
	                          cconv<2>(gid) * conv((4.3 + 7i)) +
	                              cconv<1>(gid) * conv((11.8 + 3i)));
	},
	"axpby");
auto abs = std::make_pair(
	[](double gid) {
		return std::make_pair(std::abs(rconv<1>(gid) - 4),
	                          std::abs(cconv<1>(gid) - conv((4. + 4i))));
	},
	"abs");

template<class FN, class M, class... FRS>
int run(FN && fn, M & m, FRS &&... frs) {
	return test<check_f<FN>, flecsi::mpi>(fn, m, frs...);
}

int vectest() {
	init_mesh(32, msh, coloring);
	execute<init_fields>(
		msh, xd(msh), yd(msh), zd(msh), xd_c(msh), yd_c(msh), zd_c(msh));

	UNIT () {
		vec::mesh x(msh, xd(msh)), y(msh, yd(msh)), z(msh, zd(msh)),
			tmp(msh, tmpd(msh));
		vec::mesh x_c(msh, xd_c(msh)), y_c(msh, yd_c(msh)), z_c(msh, zd_c(msh)),
			tmp_c(msh, tmpd_c(msh));

		tmp.add(x, z);
		tmp_c.add(x_c, z_c);
		EXPECT_EQ((run(add, msh, tmpd(msh), tmpd_c(msh))), 0);

		tmp.subtract(x, z);
		tmp_c.subtract(x_c, z_c);
		EXPECT_EQ((run(subtract, msh, tmpd(msh), tmpd_c(msh))), 0);

		tmp.multiply(x, z);
		tmp_c.multiply(x_c, z_c);
		EXPECT_EQ((run(mult, msh, tmpd(msh), tmpd_c(msh))), 0);

		x.add_scalar(x, 1);
		x_c.add_scalar(x_c, (1. + 1i));
		EXPECT_EQ((run(scalar_add, msh, xd(msh), xd_c(msh))), 0);

		tmp.divide(y, x);
		tmp_c.divide(y_c, x_c);
		EXPECT_EQ((run(divide, msh, tmpd(msh), tmpd_c(msh))), 0);

		x.add_scalar(x, -1);
		x_c.add_scalar(x_c, -1. - 1i);

		tmp.scale(2, x);
		tmp_c.scale(2.4, x_c);
		EXPECT_EQ((run(scale, msh, tmpd(msh), tmpd_c(msh))), 0);

		y.add_scalar(y, 1);
		y_c.add_scalar(y_c, 1. + 1i);
		tmp.reciprocal(y);
		tmp_c.reciprocal(y_c);
		EXPECT_EQ((run(recip, msh, tmpd(msh), tmpd_c(msh))), 0);

		y.add_scalar(y, -1);
		y_c.add_scalar(y_c, (-1. - 1i));

		tmp.linear_sum(8, y, 9, z);
		tmp_c.linear_sum(8, y_c, 9, z_c);
		EXPECT_EQ((run(linsum, msh, tmpd(msh), tmpd_c(msh))), 0);

		tmp.axpy(7, x, y);
		tmp_c.axpy(7. + 3i, x_c, y_c);
		EXPECT_EQ((run(axpy, msh, tmpd(msh), tmpd_c(msh))), 0);

		tmp.copy(y);
		tmp_c.copy(y_c);
		tmp.axpby(4, 11, z);
		tmp_c.axpby(4.3 + 7i, 11.8 + 3i, z_c);
		EXPECT_EQ((run(axpby, msh, tmpd(msh), tmpd_c(msh))), 0);

		tmp.add_scalar(y, -4);
		tmp_c.add_scalar(y_c, (-4. - 4i));
		tmp.abs(tmp);
		tmp_c.abs(tmp_c);
		EXPECT_EQ((run(abs, msh, tmpd(msh), tmpd_c(msh))), 0);

		tmp.add_scalar(y, -7);
		EXPECT_EQ(tmp.min().get(), -7);

		EXPECT_EQ(z.max().get(), 93);
// TODO: build errors and testing fails when using kokkos
#if !defined(FLECSI_ENABLE_KOKKOS)
		tmp.add_scalar(z, -43);
		tmp_c.add_scalar(z_c, (-37. - 43i));
		EXPECT_LT(std::abs(tmp_c.dot(x_c).get() -
		                   (-15956.319999999996 + 4052.3199999999997i)),
		          ftol);
		EXPECT_LT(std::abs(tmp_c.inf_norm().get() - 56.72741841473134), ftol);
		EXPECT_LT(std::abs(tmp_c.l1norm().get() - 1504.8788073375342), ftol);
		EXPECT_EQ(tmp.l1norm().get(), 772);
		EXPECT_EQ(tmp.inf_norm().get(), 50);
		EXPECT_EQ(tmp.dot(y).get(), 19840);
		EXPECT_EQ(tmp.global_size().get(), 32);
		EXPECT_EQ(tmp.local_size(), 32 / 4);
		EXPECT_LT(std::abs(x.l2norm().get() - 102.05880657738459), ftol);
		EXPECT_LT(std::abs(tmp_c.l2norm().get() - 268.0152234482213), ftol);
#endif
	};
}

flecsi::util::unit::driver<vectest> driver;

}
