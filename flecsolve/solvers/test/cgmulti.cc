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

using full_op = op::core<csr_op, op::shared_storage>;

template<auto var>
struct test_op : op::base<std::nullptr_t, variable_t<var>, variable_t<var>> {

	explicit test_op(full_op op) : op(op) {}

	template<class domain_vec, class range_vec>
	void apply(const domain_vec & x, range_vec & y) const {
		op.apply(x, y);
	}

protected:
	full_op op;
};

int multicg() {
	UNIT () {
		auto mtx = mat::io::matrix_market<>::read("Chem97ZtZ.mtx").tocsr();

		init_mesh(mtx.rows(), msh, coloring);
		full_op A(std::move(mtx));

		auto xm = vec::make(vec::make(variable<vars::var1>, msh, xmd[0](msh)),
		                    vec::make(variable<vars::var2>, msh, xmd[1](msh)));
		auto bm = vec::make(vec::make(variable<vars::var1>, msh, bmd[0](msh)),
		                    vec::make(variable<vars::var2>, msh, bmd[1](msh)));

		bm.set_scalar(0.0);
		bm.subset(variable<vars::var2>).set_random(3);
		xm.subset(variable<vars::var1>).set_random(7);
		xm.subset(variable<vars::var2>).set_random(4);

		auto settings = read_config("cgmulti.cfg", cg::options("solver"));

		op::krylov slv1(op::krylov_parameters(
			settings,
			cg::topo_work<>::get(bm.subset(variable<vars::var1>)),
			op::core<test_op<vars::var1>>(A)));
		auto info1 = slv1.apply(bm, xm);

		op::krylov slv2(op::krylov_parameters(
			settings,
			cg::topo_work<>::get(bm.subset(variable<vars::var2>)),
			op::core<test_op<vars::var2>>(A)));
		auto info2 = slv2.apply(bm, xm);

		EXPECT_EQ(info1.iters, 161);
		EXPECT_EQ(info2.iters, 143);
	};

	return 0;
}

flecsi::util::unit::driver<multicg> driver;

}
