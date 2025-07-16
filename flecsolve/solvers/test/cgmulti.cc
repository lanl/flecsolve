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

const realf::definition<testmesh, testmesh::cells> xd, bd;

enum class vars { var1, var2 };
const std::array<realf::definition<testmesh, testmesh::cells>, 2> xmd, bmd;

template<auto V>
auto make_test_op(variable_t<V>, const mat::csr<double> & m) {
	return op::core<csr_op_gen<variable_t<V>, variable_t<V>>>(m);
}

int multicg(flecsi::scheduler & s) {
	UNIT () {
		testmesh::ptr mptr;

		auto mtx = mat::io::matrix_market<>::read("Chem97ZtZ.mtx").tocsr();

		auto & msh = init_mesh(s, mtx.rows(), mptr);

		auto xm = vec::make(vec::make(variable<vars::var1>, xmd[0](msh)),
		                    vec::make(variable<vars::var2>, xmd[1](msh)));
		auto bm = vec::make(vec::make(variable<vars::var1>, bmd[0](msh)),
		                    vec::make(variable<vars::var2>, bmd[1](msh)));

		bm.set_scalar(0.0);
		bm.subset(variable<vars::var2>).set_random(3);
		xm.subset(variable<vars::var1>).set_random(7);
		xm.subset(variable<vars::var2>).set_random(4);

		auto settings = read_config("cgmulti.cfg", cg::options("solver"));

		auto op1 = make_test_op(variable<vars::var1>, mtx);
		auto op2 = make_test_op(variable<vars::var2>, mtx);

		auto slv1 = cg::solver(
			settings,
			cg::make_work(bm.subset(variable<vars::var1>)))(op::ref(op1));
		auto info1 = slv1(bm, xm);

		auto slv2 = cg::solver(
			settings,
			cg::make_work(bm.subset(variable<vars::var2>)))(op::ref(op2));
		auto info2 = slv2(bm, xm);

		EXPECT_EQ(info1.iters, 161);
		EXPECT_EQ(info2.iters, 143);
	};

	return 0;
}

flecsi::util::unit::driver<multicg> driver;

}
