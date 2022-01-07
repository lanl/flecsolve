#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsi-linalg/vectors/mesh.hh"

#include "test_mesh.hh"

using namespace flecsi;

namespace flecsi::linalg {

testmesh::slot msh;
testmesh::cslot coloring;

const field<double>::definition<testmesh, testmesh::cells> xd, yd, zd, tmpd;

void init_mesh() {
	std::vector<std::size_t> extents{32};
	auto colors = testmesh::distribute(flecsi::processes(), extents);
	coloring.allocate(colors, extents);

	msh.allocate(coloring.get());
}

void init_fields(testmesh::accessor<ro, ro> m,
                 field<double>::accessor<wo, na> xa,
                 field<double>::accessor<wo, na> ya,
                 field<double>::accessor<wo, na> za) {
	for (auto dof : m.dofs<testmesh::cells>()) {
		xa[dof] = m.global_id(dof);
		ya[dof] = m.global_id(dof) * 2;
		za[dof] = m.global_id(dof) * 3;
	}
}

template <class F, class T>
struct check {
	static constexpr double ftol = 1e-8;

	int operator()(testmesh::accessor<ro, ro> m,
	               field<double>::accessor<ro, na> x) {
		UNIT(name) {
			for (auto dof : m.dofs<testmesh::cells>()) {
				auto gid = m.global_id(dof);
				EXPECT_LT(std::abs(f(gid) - x[dof]), ftol);
			}
		};
	}

	F f;
	T name;
};
template <class F, class T>
check(F&&,T&&)->check<F,T>;


int vectest() {
	init_mesh();
	execute<init_fields>(msh, xd(msh), yd(msh), zd(msh));

	UNIT() {
		vec::mesh x(msh, xd(msh)), y(msh, yd(msh)), z(msh, zd(msh)), tmp(msh, tmpd(msh));

		tmp.add(x, z);
		static check add{[](std::size_t gid) {
			return gid + 3 * gid;
		}, "add"};
		EXPECT_EQ((test<add>(msh, tmpd(msh))), 0);

		tmp.subtract(x, z);
		static check subtract{[](int gid) {
			return gid - 3 * gid;
		}, "subtract"};
		EXPECT_EQ((test<subtract>(msh, tmpd(msh))), 0);

		tmp.multiply(x, z);
		static check mult{[](std::size_t gid) {
			return gid * 3 * gid;
		}, "multiply"};
		EXPECT_EQ((test<mult>(msh, tmpd(msh))), 0);

		x.add_scalar(x, 1);
		static check scalar_add{[](std::size_t gid) {
			return gid + 1;
		}, "add scalar"};
		EXPECT_EQ((test<scalar_add>(msh, xd(msh))), 0);

		tmp.divide(y, x);
		static check divide{[](double gid) {
			return (gid*2) / (gid + 1);
		}, "divide"};
		EXPECT_EQ((test<divide>(msh, tmpd(msh))), 0);

		x.add_scalar(x, -1);

		tmp.scale(2, x);
		static check scale{[](std::size_t gid) {
			return gid * 2;
		}, "scale"};
		EXPECT_EQ((test<scale>(msh, tmpd(msh))), 0);

		y.add_scalar(y, 1);
		tmp.reciprocal(y);
		static check recip{[](double gid) {
			return 1.0 / (2*gid + 1);
		}, "reciprocal"};
		EXPECT_EQ((test<recip>(msh, tmpd(msh))), 0);
		y.add_scalar(y, -1);

		tmp.linear_sum(8, y, 9, z);
		static check linsum{[](std::size_t gid) {
			return (2*gid)*8 + 9*(3*gid);
		}, "linear sum"};
		EXPECT_EQ((test<linsum>(msh, tmpd(msh))), 0);

		tmp.axpy(7, x, y);
		static check axpy{[](double gid) {
			return 7 * gid + (gid * 2);
		}, "axpy"};
		EXPECT_EQ((test<axpy>(msh, tmpd(msh))), 0);

		tmp.copy(y);
		tmp.axpby(4, 11, z);
		static check axpby{[](double gid) {
			return 4 * (3*gid) + 11 * (2*gid);
		}, "axpby"};
		EXPECT_EQ((test<axpby>(msh, tmpd(msh))), 0);

		tmp.add_scalar(y, -4);
		tmp.abs(tmp);
		static check abs{[](double gid) {
			return std::abs(2*gid - 4);
		}, "abs"};
		EXPECT_EQ((test<abs>(msh, tmpd(msh))), 0);

		tmp.add_scalar(y, -7);
		EXPECT_EQ(tmp.min().get(), -7);

		auto v = z.max().get();
		EXPECT_EQ(z.max().get(), 93);

		tmp.add_scalar(z, -43);
		EXPECT_EQ(tmp.l1norm().get(), 772);
		EXPECT_EQ(tmp.inf_norm().get(), 138);
		EXPECT_EQ(tmp.inner_prod(y).get(), 19840);
		EXPECT_EQ(tmp.global_size().get(), 32);
		EXPECT_EQ(tmp.local_size(), 32/4);
		EXPECT_LT(std::abs(x.l2norm().get() - 102.05880657738459), add.ftol);
	};
}


unit::driver<vectest> driver;

}
