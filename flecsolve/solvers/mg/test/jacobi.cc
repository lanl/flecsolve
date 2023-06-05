#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/solvers/mg/jacobi.hh"

namespace flecsolve {

using namespace flecsi;

using csr = topo::csr<double>;
csr::vec_def<csr::cols> ud, fd;

namespace {

int jacobitest() {

	UNIT () {
		using parcsr = mat::parcsr<double>;
		parcsr A{parcsr::parameters(
			MPI_COMM_WORLD, flecsi::processes(), "diag-diff.mtx")};
		auto u = A.vec(ud);
		auto f = A.vec(fd);

		u.set_random();
		f.zero();

		mg::jacobi relax{mg::jacobi_params{std::ref(A), 2 / 3., 50}};
		relax.apply(f, u);
	};
	return 0;
}

flecsi::util::unit::driver<jacobitest> driver;
}
}
