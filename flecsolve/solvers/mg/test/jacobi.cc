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

int jacobitest(flecsi::scheduler & s) {

	UNIT () {
		op::core<parcsr> A(s, MPI_COMM_WORLD, "nos7.mtx");
		auto Ah = op::ref(A);
		auto & topo = A.data.topo();
		auto [u, f] = vec::make(topo)(ud, fd);

		u.set_random();
		f.zero();

		mg::jacobi smoother{
			read_config("jacobi.cfg", mg::jacobi::options("smoother"))};
		smoother(Ah)(f, u);
	};
	return 0;
}

flecsi::util::unit::driver<jacobitest> driver;
}
}
