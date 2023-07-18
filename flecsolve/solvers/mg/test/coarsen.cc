#include "flecsi/flog.hh"

#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/matrices/parcsr.hh"
#include "flecsolve/solvers/mg/coarsen.hh"

namespace flecsolve {

using namespace flecsi;

using scalar = double;

using csr = topo::csr<scalar>;
using parcsr = mat::parcsr<scalar>;

namespace {

int coarsentest() {
	UNIT () {
		parcsr A{parcsr::parameters{
			MPI_COMM_WORLD, flecsi::processes(), "poisson.mtx"}};
		mg::ua::coarsen(A);
	};
	return 0;
}

flecsi::util::unit::driver<coarsentest> driver;
}

}
