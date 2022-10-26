#include <cmath>
#include <array>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/vectors/multi.hh"
#include "flecsolve/operators/base.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/util/config.hh"

#include "csr_utils.hh"

namespace flecsolve {

testmesh::slot msh;
testmesh::cslot coloring;

const realf::definition<testmesh, testmesh::cells> xd, bd;

enum class vars { var1, var2 };
const std::array<realf::definition<testmesh, testmesh::cells>, 2> xmd, bmd;

template<auto var, class Op>
struct test_op : op::base<test_op<var, Op>> {
	template<auto V>
	test_op(variable_t<V>, const Op & op) : op(op) {}

	template<class domain_vec, class range_vec>
	void apply(const domain_vec & x, range_vec & y) const {
		op.apply(x, y);
	}

protected:
	const Op & op;
};
template<auto V, class Op>
test_op(variable_t<V>, const Op &) -> test_op<V, Op>;

namespace op {
template<auto var, class Op>
struct traits<test_op<var, Op>> {
	static constexpr auto input_var = variable<var>;
	static constexpr auto output_var = variable<var>;
};
}

int multicg() {
	UNIT () {
		auto mat = read_mm("Chem97ZtZ.mtx");

		init_mesh(mat.nrows, msh, coloring);
		csr_op A{std::move(mat)};

		vec::multi xm(vec::mesh(variable<vars::var1>, msh, xmd[0](msh)),
		              vec::mesh(variable<vars::var2>, msh, xmd[1](msh)));
		vec::multi bm(vec::mesh(variable<vars::var1>, msh, bmd[0](msh)),
		              vec::mesh(variable<vars::var2>, msh, bmd[1](msh)));

		bm.set_scalar(0.0);
		bm.subset(variable<vars::var2>).set_random(3);
		xm.subset(variable<vars::var1>).set_random(7);
		xm.subset(variable<vars::var2>).set_random(4);

		test_op A1(variable<vars::var1>, A);
		test_op A2(variable<vars::var2>, A);

		cg::settings settings("solver");
		read_config("cgmulti.cfg", settings);

		op::krylov slv1(op::krylov_parameters(
			settings,
			cg::topo_work<>::get(bm.subset(variable<vars::var1>)),
			std::move(A1)));
		auto info1 = slv1.apply(bm, xm);

		op::krylov slv2(op::krylov_parameters(
			settings,
			cg::topo_work<>::get(bm.subset(variable<vars::var2>)),
			std::move(A2)));
		auto info2 = slv2.apply(bm, xm);

		EXPECT_EQ(info1.iters, 161);
		EXPECT_EQ(info2.iters, 143);
	};

	return 0;
}

flecsi::unit::driver<multicg> driver;

}
