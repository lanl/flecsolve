#include <array>

#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/vectors/multi.hh"


#include "test_mesh.hh"

using namespace flecsi;

namespace flecsi::linalg {

testmesh::slot msh;
testmesh::cslot coloring;

constexpr std::size_t nvars = 4;
using fd_array = std::array<field<double>::definition<testmesh, testmesh::cells>, nvars>;
const fd_array xd, yd, zd, tmpd;
using make_is = std::make_index_sequence<nvars>;

void init_mesh() {
	std::vector<std::size_t> extents{32};
	auto colors = testmesh::distribute(flecsi::processes(), extents);
	coloring.allocate(colors, extents);

	msh.allocate(coloring.get());
}


void init_field(testmesh::accessor<ro, ro> m,
                field<double>::accessor<wo, na> xa,
                int offset,
                std::size_t index) {
	for (auto dof : m.dofs<testmesh::cells>()) {
		xa[dof] = (offset+index+1)*m.global_id(dof);
	}
}

template <std::size_t ... Index>
void init_fields(const fd_array & arr, int offset, std::index_sequence<Index...>) {
	(execute<init_field>(msh, arr[Index](msh), offset, Index),...);
}

template <std::size_t... Index>
auto create_multivector(const fd_array & arr, std::index_sequence<Index...>) {
	return vec::multi(vec::mesh(msh, arr[Index](msh))...);
}


template <class F, class T>
struct check {
	static constexpr double ftol = 1e-8;

	int operator()(testmesh::accessor<ro, ro> m,
	               field<double>::accessor<ro, na> x,
	               int index) {
		UNIT(name) {
			for (auto dof : m.dofs<testmesh::cells>()) {
				auto gid = m.global_id(dof);
				EXPECT_LT(std::abs(f(gid,index) - x[dof]), ftol);
			}
		};
	}

	F f;
	T name;
};
template <class F, class T>
check(F&&,T&&)->check<F,T>;


template<auto & F, class MV, class S>
bool run(MV & mv, S & msh) {
	int ind = 0;
	return std::apply([&](auto & ... v) {
		return ((test<F>(msh, v.data.ref(), ind++) == 0) and ...);
	}, mv.data);
}


int vectest() {
	init_mesh();
	init_fields(xd, 0, make_is());
	init_fields(yd, 1, make_is());
	init_fields(zd, 2, make_is());
	UNIT() {
		auto x = create_multivector(xd, make_is());
		auto y = create_multivector(yd, make_is());
		auto z = create_multivector(zd, make_is());
		auto tmp = create_multivector(tmpd, make_is());

		tmp.add(x, z);
		static check add{[](std::size_t gid, int index) {
			return (index + 1)*gid + (index + 3)*gid;
		}, "add"};
		EXPECT_TRUE(run<add>(tmp, msh));

		tmp.subtract(y, z);
		static check sub{[](double gid, double index) {
			return (index + 2)*gid - (index + 3)*gid;
		}, "subtract"};
		EXPECT_TRUE(run<sub>(tmp, msh));

		tmp.multiply(x, z);
		static check mult{[](std::size_t gid, int index) {
			return (index + 1)*gid * (index + 3)*gid;
		}, "multiply"};
		EXPECT_TRUE(run<mult>(tmp, msh));

		x.add_scalar(x, 1);
		static check scalar_add{[](std::size_t gid, int index) {
			return (index+1)*gid + 1;
		}, "add scalar"};
		EXPECT_TRUE(run<scalar_add>(x, msh));

		tmp.divide(y, x);
		static check divide{[](double gid, double index) {
			return ((index+2)*gid) / ((index+1)*gid + 1);
		}, "divide"};
		EXPECT_TRUE(run<divide>(tmp, msh));

		x.add_scalar(x, -1);

		tmp.scale(3.5, x);
		static check scale{[](double gid, double index) {
			return (index+1)*gid * 3.5;
		}, "scale"};
		EXPECT_TRUE(run<scale>(tmp, msh));

		y.add_scalar(y, 2);
		tmp.reciprocal(y);
		static check recip{[](double gid, double index) {
			return 1.0 / ((index+2)*gid + 2);
		}, "reciprocal"};
		EXPECT_TRUE(run<recip>(tmp, msh));
		y.add_scalar(y, -2);

		tmp.linear_sum(8, y, 9, z);
		static check linsum{[](double gid, double index) {
			return ((index+2)*gid) * 8 + ((index + 3)*gid)*9;
		}, "linear sum"};
		EXPECT_TRUE(run<linsum>(tmp, msh));

		tmp.axpy(7, x, y);
		static check axpy{[](double gid, double index) {
			return (index+1)*gid*7 + ((index+2)*gid);
		}, "axpy"};
		EXPECT_TRUE(run<axpy>(tmp, msh));

		tmp.copy(y);
		tmp.axpby(4, 11, z);
		static check axpby{[](double gid, double index) {
			return ((index+3)*gid)*4 + ((index+2)*gid)*11;
		}, "axpby"};
		EXPECT_TRUE(run<axpby>(tmp, msh));

		tmp.add_scalar(y, -4.3);
		tmp.abs(tmp);
		static check abs{[](double gid, double index) {
			return std::abs((index+2)*gid - 4.3);
		}, "abs"};
		EXPECT_TRUE(run<abs>(tmp, msh));

		tmp.add_scalar(y, -7);
		EXPECT_EQ(tmp.min().get(), -7);

		{
			auto & [t0, t1, t2, t3] = tmp;
			EXPECT_EQ(tmp.max().get(),
			          std::max({t0.max().get(), t1.max().get(), t2.max().get(), t3.max().get()}));

			tmp.add_scalar(z, -43);
			EXPECT_EQ(tmp.l1norm().get(),
			          t0.l1norm().get() + t1.l1norm().get() + t2.l1norm().get() + t3.l1norm().get());
			EXPECT_EQ(tmp.l2norm().get(),
			          std::sqrt(std::pow(t0.l2norm().get(),2) + std::pow(t1.l2norm().get(),2) +
			                    std::pow(t2.l2norm().get(),2) + std::pow(t3.l2norm().get(),2)));
			EXPECT_EQ(tmp.inf_norm().get(),
			          std::max({t0.inf_norm().get(), t1.inf_norm().get(), t2.inf_norm().get(), t3.inf_norm().get()}));
		}

		{ // inner product
			auto & [x0, x1, x2, x3] = x;
			auto & [y0, y1, y2, y3] = y;
			EXPECT_EQ(x.inner_prod(y).get(),
			          (x0.inner_prod(y0).get() +
			           x1.inner_prod(y1).get() +
			           x2.inner_prod(y2).get() +
			           x3.inner_prod(y3).get()));
		}
	};
}

unit::driver<vectest> driver;

}
