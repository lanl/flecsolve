#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/solvers/mg/jacobi.hh"

namespace flecsolve {

using namespace flecsi;

using csr = topo::csr<double>;
csr::vec_def<csr::cols> ud, fd;

static void set_problem(csr::accessor<ro> A,
                        field<double>::accessor<wo, na> f) {
	constexpr double pi = M_PI;
	auto rhs = [](double x, double y) {
		return 8 * (pi * pi) * sin(2 * pi * x) * sin(2 * pi * y);
	};
	constexpr std::size_t n = 102;
	constexpr double h = 1. / (n - 1);
	for (auto dof : A.dofs<csr::cols>()) {
		auto cid = A.global_id(dof);
		const auto ix = (cid % (n - 2)) + 1;
		const auto iy = (cid / (n - 2)) + 1;
		const float x = ix * h;
		const float y = iy * h;
		f[dof] = (h * h) * rhs(x, y);
	}
}

int jacobitest() {

	UNIT () {
		using namespace flecsolve::mat;
		parcsr<double> A{
			parcsr_params{MPI_COMM_WORLD, flecsi::processes(), "poisson.mtx"}};
		auto u = A.vec(ud);
		auto f = A.vec(fd);

		u.set_random();
		execute<set_problem>(A.data.topo(), f.data.ref());

		mg::jacobi relax{mg::jacobi_params{std::ref(A), 1 / 3., 10000}};
		relax.apply(f, u);
	};
	return 0;
}

flecsi::util::unit::driver<jacobitest> driver;
}
