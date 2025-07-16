#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>
#include <iomanip>

#include "flecsolve/topo/csr.hh"
#include "flecsolve/vectors/seq.hh"
#include "flecsolve/operators/core.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/matrices/parcsr.hh"

namespace flecsolve {

using namespace flecsi;

using parcsr = mat::parcsr<double>;
using csr_topo = parcsr::topo_t;
csr_topo::vec_def<csr_topo::cols> xd, yd;

int csr_test(flecsi::scheduler & s) {
	UNIT () {
		op::core<parcsr> A(s, MPI_COMM_WORLD, "Chem97ZtZ.mtx");
		auto & topo = A.data.topo();
		auto [x, y] = vec::make(topo)(xd, yd);
		y.set_scalar(0.0);
		x.set_scalar(2);

		auto slv = cg::solver(
			read_config("parcsr.cfg", cg::options("solver")),
			cg::make_work(x))(op::ref(A));

		auto info = slv(y, x);
		EXPECT_TRUE(info.iters == 167);
	};
	return 0;
}

flecsi::util::unit::driver<csr_test> driver;
}
