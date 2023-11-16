#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/matrices/io/matrix_market.hh"
#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/vectors/seq.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/util/config.hh"
#include "flecsolve/vectors/traits.hh"

namespace flecsolve {

template<class Op, class Vec>
struct diagnostic {
	diagnostic(const Op & A, const Vec & x0, double cond)
		: iter(0), cond(cond), A(A), Ax{x0.data.size()}, monotonic_fail(false),
		  convergence_fail(false) {
		A.apply(x0, Ax);
		auto nrm = x0.dot(Ax).get();
		e_0 = std::sqrt(nrm);
		e_prev = e_0;
	}

	template<class T, std::enable_if_t<is_vector_v<T>, bool> = true>
	bool operator()(const T & x, double) {
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

	int iter;
	double e_prev;
	double e_0;
	double cond;
	const Op & A;
	Vec Ax;

	bool monotonic_fail;
	bool convergence_fail;
};

struct tcase {
	const char * fname; // filename
	float cond; // condition number
	int iters; // expected iterations to converge
	int iters_rel;
};
int cgtest() {
	std::array cases{tcase{"494_bus.mtx", 2.415411e+06, 1822, 1829},
	                 tcase{"Chem97ZtZ.mtx", 2.472189e+02, 161, 161}};

	UNIT () {
		for (const auto & cs : cases) {
			op::core<mat::csr<double>, op::shared_storage> A(
				mat::io::matrix_market<>::read(cs.fname).tocsr());

			vec::seq_vec<double> b{A.source().rows()};
			vec::seq_vec<double> x{A.source().rows()};

			b.set_scalar(0.0);
			x.set_random(7);

			diagnostic diag(A, x, cs.cond);
			cg::settings settings{"cg-solver"};
			read_config("cg.cfg", settings);
			op::krylov slv(
				op::krylov_parameters(std::move(settings),
			                          vec::seq_work<double, cg::nwork>{b},
			                          A,
			                          op::I,
			                          std::ref(diag)));

			auto info = slv.apply(b, x);

			EXPECT_TRUE((info.iters == cs.iters) ||
			            (info.iters == cs.iters_rel));
			EXPECT_FALSE(diag.monotonic_fail);
			EXPECT_FALSE(diag.convergence_fail);
			EXPECT_TRUE(info.iters == diag.iter);
		}
	};

	return 0;
}

flecsi::util::unit::driver<cgtest> driver;

}
