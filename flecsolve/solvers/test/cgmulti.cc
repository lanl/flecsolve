#include <cmath>
#include <array>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/vectors/multi.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/util/config.hh"
#include "flecsolve/matrices/io/matrix_market.hh"

#include "flecsolve/util/test/mesh.hh"

namespace flecsolve {

testmesh::slot msh;
testmesh::cslot coloring;

const realf::definition<testmesh, testmesh::cells> xd, bd;

enum class vars { var1, var2 };
const std::array<realf::definition<testmesh, testmesh::cells>, 2> xmd, bmd;

template<auto var, class Op>
struct test_op : op::base<std::nullptr_t, variable_t<var>, variable_t<var>> {
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

int multicg() {
	UNIT () {
		auto mtx = mat::io::matrix_market<>::read("Chem97ZtZ.mtx").tocsr();

		init_mesh(mtx.rows(), msh, coloring);
		auto A = op::make(csr_op{std::move(mtx)});

		vec::multi xm(vec::topo_view(variable<vars::var1>, msh, xmd[0](msh)),
		              vec::topo_view(variable<vars::var2>, msh, xmd[1](msh)));
		vec::multi bm(vec::topo_view(variable<vars::var1>, msh, bmd[0](msh)),
		              vec::topo_view(variable<vars::var2>, msh, bmd[1](msh)));

		bm.set_scalar(0.0);
		bm.subset(variable<vars::var2>).set_random(3);
		xm.subset(variable<vars::var1>).set_random(7);
		xm.subset(variable<vars::var2>).set_random(4);

		auto A1 = op::make(test_op(variable<vars::var1>, A));
		auto A2 = op::make(test_op(variable<vars::var2>, A));

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

flecsi::util::unit::driver<multicg> driver;

}
