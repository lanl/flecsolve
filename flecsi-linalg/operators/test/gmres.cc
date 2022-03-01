#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/operators/gmres.hh"


#include "csr_utils.hh"


namespace flecsi::linalg {

static constexpr std::size_t ncases = 2;
std::array<testmesh::slot, ncases> mshs;
std::array<testmesh::cslot, ncases> colorings;

const realf::definition<testmesh, testmesh::cells> xd, bd;

template <class Op, class Vec>
struct diagnostic {
	diagnostic(const Op & A, const Vec & x0, const Vec & b, double cfact) :
		iter(0), cfact(cfact), fail_monotonic(false),
		fail_convergence(false) {
		Vec r(x0.data.topo, resdef(x0.data.topo));
		A.residual(b, x0, r);
		rnorm0 = r.l2norm().get();
		rnorm_prev = rnorm0;
	}

	bool operator()(const Vec &, double rnorm) {
		float n = ++iter;
		auto bnd = std::pow(cfact, n / 2) * rnorm0;

		fail_monotonic = fail_monotonic || (rnorm > rnorm_prev);
		fail_convergence = fail_convergence || (rnorm > bnd);

		rnorm_prev = rnorm;

		return false;
	}

	std::size_t iter;
	double rnorm0, rnorm_prev;
	double cfact;

	bool fail_monotonic;
	bool fail_convergence;

	static inline const realf::definition<testmesh, testmesh::cells> resdef;
};


int gmres_test() {
	UNIT() {
		auto mat = read_mm("Chem97ZtZ.mtx");

		double cond = 2.472189e+02;
		double cfact = (cond*cond - 1) / (cond * cond);

		auto & msh = mshs[0];
		init_mesh(mat.nrows, msh, colorings[0]);
		csr_op A{std::move(mat)};

		vec::mesh x(msh, xd(msh)), b(msh, bd(msh));

		b.set_random();
		x.set_random();

		diagnostic diag(A, x, b, cfact);
		gmres::solver slv(gmres::default_settings(op::I, diag),
		                  gmres::topo_work<>::get(b));


		slv.apply(A, b, x);

		EXPECT_FALSE(diag.fail_monotonic);
		EXPECT_FALSE(diag.fail_convergence);
	};

	return 0;
}


unit::driver<gmres_test> driver;

}
