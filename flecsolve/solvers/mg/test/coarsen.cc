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
csr::vec_def<csr::cols> aggt_def;

namespace {

int coarsentest() {
	UNIT () {
		parcsr A{parcsr::parameters{
			MPI_COMM_WORLD, flecsi::processes(), "diag-diff-50.mtx"}};
		auto Ac = mg::ua::coarsen(A, aggt_def(A.data.topo()));
		flecsi::execute<dump, flecsi::mpi>(A.data.topo(), "fine");
		flecsi::execute<dump, flecsi::mpi>(Ac.data.topo(), "coarse");
		(void)Ac;
	};
	return 0;
}

flecsi::util::unit::driver<coarsentest> driver;
}

}
