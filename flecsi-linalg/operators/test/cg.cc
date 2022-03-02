#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"


#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/operators/cg.hh"


#include "csr_utils.hh"

namespace flecsi::linalg {

static constexpr std::size_t ncases = 2;
std::array<testmesh::slot, ncases> mshs;
std::array<testmesh::cslot, ncases> colorings;

const realf::definition<testmesh, testmesh::cells> xd, bd;



template <class Op, class Vec>
struct diagnostic
{
	diagnostic(const Op & A, const Vec & x0, double cond) :
		iter(0), cond(cond), A(A),
		Ax(x0.data.topo, axdef(x0.data.topo)),
		monotonic_fail(false),
		convergence_fail(false)
	{
		A.apply(x0, Ax);
		auto nrm = x0.dot(Ax).get();
		e_0 = std::sqrt(nrm);
		e_prev = e_0;
	}

	bool operator()(const Vec & x, double) {
		A.apply(x, Ax);
		auto nrm = x.dot(Ax).get();
		auto e_a = std::sqrt(nrm);
		auto frac = (std::sqrt(cond) - 1) / (std::sqrt(cond) + 1);
		auto bnd = 2 * std::pow(frac, ++iter) * e_0;

		// Check || e_k ||_A monotonically decreases
		monotonic_fail = monotonic_fail || (e_a > e_prev);
		// Convergence check (Trefethen and Bau NLA, p. 299)
		convergence_fail = convergence_fail || (e_a >= bnd);

		e_prev = e_a;

		return false;
	}

	std::size_t iter;
	double e_prev;
	double e_0;
	double cond;
	const Op & A;
	Vec Ax;

	bool monotonic_fail;
	bool convergence_fail;
	static inline const realf::definition<testmesh, testmesh::cells> axdef;
};


int cgtest() {
	// pair is filename, condition number
	std::array cases{
		std::make_pair("494_bus.mtx", 2.415411e+06),
		std::make_pair("Chem97ZtZ.mtx", 2.472189e+02)
	};

	static_assert(cases.size() <= ncases);

	UNIT() {
		std::size_t i = 0;
		for (const auto & cs : cases) {
			auto mat = read_mm(cs.first);

			auto & msh = mshs[i];

			init_mesh(mat.nrows, msh, colorings[i]);
			csr_op A{std::move(mat)};

			vec::mesh x(msh, xd(msh)), b(msh, bd(msh));
			b.set_scalar(0.0);
			x.set_random();

			diagnostic diag(A, x, cs.second);
			cg::solver slv(cg::default_settings(),
			               cg::topo_work<>::get(b));

			slv.settings.maxiter = 2000;
			slv.bind(op::I, diag).apply(A, b, x);

			EXPECT_FALSE(diag.monotonic_fail);
			EXPECT_FALSE(diag.convergence_fail);

			++i;
		}
	};


	return 0;
}

unit::driver<cgtest> driver;

}

