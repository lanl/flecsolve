#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>
#include <iomanip>

#include "flecsolve/vectors/seq.hh"
#include "flecsolve/operators/core.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/krylov_operator.hh"
#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/matrices/parcsr.hh"

namespace flecsolve {

using namespace flecsi;

using parcsr = mat::parcsr<double>;
using csr_topo = parcsr::topo_t;
csr_topo::vec_def<csr_topo::cols> xd, yd;

int csr_test() {
	UNIT () {
		op::core<parcsr> A(MPI_COMM_WORLD, "Chem97ZtZ.mtx");
		auto & topo = A.data.topo();
		auto [x, y] = vec::make(topo)(xd, yd);

		y.set_scalar(0.0);
		x.set_scalar(2);

		auto slv = op::krylov_solver(op::krylov_parameters(
			read_config("parcsr.cfg", cg::options("solver")),
			cg::topo_work<>::get(x),
			std::ref(A)));

		auto info = slv.apply(y, x);
		EXPECT_TRUE(info.iters == 167);
	};
	return 0;
}

flecsi::util::unit::driver<csr_test> driver;
}
