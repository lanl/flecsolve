#ifndef FLECSOLVE_SOLVERS_MG_CYCLE_HH
#define FLECSOLVE_SOLVERS_MG_CYCLE_HH

#include <cstddef>

namespace flecsolve::mg {

template<class Domain, class Range, class Hier, class CoarseSolver>
void ncycle(std::size_t lvl, const Domain & b, Range & x, Hier & hier,
            const CoarseSolver & coarse_solver, int n = 1) {
	auto & level = hier.get(lvl);
	auto & clevel = hier.get(lvl + 1);
	auto & A = level.A();

	level.presmoother()(b, x);

	auto & r = level.res();
	A.residual(b, x, r);

	auto & coarse_b = clevel.rhs();
	auto & coarse_x = clevel.sol();

	clevel.R()(r, coarse_b);
	coarse_x.set_scalar(0.);
	auto coarse_lvl = hier.depth() - 1;
	if (lvl + 1 == coarse_lvl) {
		coarse_solver(coarse_b, coarse_x);
	} else {
		for (int i = 0; i < n; ++i)
			ncycle(lvl + 1, coarse_b, coarse_x, hier, coarse_solver, n);
	}
	auto & correction = level.correction();
	clevel.P()(coarse_x, correction);
	x.add(x, correction);
	level.postsmoother()(b, x);
}

template<class D, class R, class Hier, class CoarseSolver>
void vcycle(const D & b, R & x,
            Hier & hier, const CoarseSolver & coarse_solver) {
	if (hier.depth() == 1) {
		coarse_solver(b, x);
	} else {
		ncycle(0, b, x, hier, coarse_solver);
	}
}

}

#endif
