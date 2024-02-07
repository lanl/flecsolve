#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/matrices/parcsr.hh"
#include "flecsolve/solvers/mg/jacobi.hh"

namespace flecsolve {

using namespace flecsi;

using parcsr = mat::parcsr<double>;
using csr_topo = parcsr::topo_t;

csr_topo::vec_def<csr_topo::cols> ud, fd;

namespace {

int jacobitest() {

	UNIT () {
		op::core<parcsr, op::shared_storage> A(MPI_COMM_WORLD, "nos7.mtx");
		auto & topo = A.source().data.topo();
		auto [u, f] = vec::make(topo)(ud, fd);

		u.set_random();
		f.zero();

		mg::jacobi smoother{
			read_config("jacobi.cfg", mg::jacobi::options("smoother"))};
		smoother(A)(f, u);
	};
	return 0;
}

flecsi::util::unit::driver<jacobitest> driver;
}
}
