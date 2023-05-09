#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>
#include <flecsi/execution.hh>
#include <iomanip>

#include "flecsolve/vectors/seq.hh"
#include "flecsolve/operators/base.hh"
#include "flecsolve/solvers/factory.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/krylov_operator.hh"
#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/matrices/io/matrix_market.hh"
#include "flecsolve/matrices/parcsr.hh"

namespace flecsolve {

using namespace flecsi;

using csr = topo::csr<double>;

csr::vec_def<csr::cols> xd;
csr::vec_def<csr::cols> yd;

int csr_test() {
	UNIT () {
		using namespace flecsolve::mat;

		auto A = io::matrix_market<double, std::size_t>::readpar(
			MPI_COMM_WORLD, "Chem97ZtZ.mtx", flecsi::processes());
		auto x = A.vec(xd);
		auto y = A.vec(yd);

		y.set_scalar(0.0);
		x.set_scalar(2);

		op::krylov_parameters params{
			cg::settings("solver"), cg::topo_work<>::get(x), std::ref(A)};
		read_config("parcsr.cfg", params);

		op::krylov slv{std::move(params)};

		auto info = slv.apply(y, x);
		EXPECT_TRUE(info.iters == 167);
	};
	return 0;
}

flecsi::util::unit::driver<csr_test> driver;
}
