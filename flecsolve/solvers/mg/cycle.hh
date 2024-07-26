#ifndef FLECSOLVE_SOLVERS_MG_CYCLE_HH
#define FLECSOLVE_SOLVERS_MG_CYCLE_HH

#include <cstddef>
#include <istream>

namespace flecsolve::mg {

enum class cycle_type { v, relaxed_w };

inline std::istream & operator>>(std::istream & in, cycle_type & ctype) {
	std::string tok;
	in >> tok;

	if (tok == "v")
		ctype = cycle_type::v;
	else if (tok == "relaxed-w")
		ctype = cycle_type::relaxed_w;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

inline std::ostream & operator<<(std::ostream & os, const cycle_type & ctype) {
	if (ctype == cycle_type::v)
		os << "v";
	else if (ctype == cycle_type::relaxed_w)
		os << "relaxed-w";

	return os;
}


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



template<class D, class R, class Hier, class CoarseSolver>
void relaxed_wcycle(std::size_t lvl, const D & b, R & x,
                    Hier & hier, const CoarseSolver & coarse_solver,
                    double tau) {
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
		auto & Ac = clevel.A();
		auto & c = clevel.c();
		auto & v = clevel.v();
		auto & btilde = clevel.btilde();
		auto & d = clevel.d();

		c.set_scalar(0.);
		relaxed_wcycle(lvl+1, coarse_b, c, hier, coarse_solver, tau);
		Ac(c, v);
		btilde.axpy(-tau, v, coarse_b);
		d.set_scalar(0.);
		relaxed_wcycle(lvl+1, btilde, d, hier, coarse_solver, tau);
		coarse_x.linear_sum(tau, c, tau, d);
	}

	auto & correction = level.correction();
	clevel.P()(coarse_x, correction);
	x.add(x, correction);
	level.postsmoother()(b, x);
}


template<class D, class R, class Hier, class CoarseSolver>
void relaxed_wcycle(const D & b, R & x,
                    Hier & hier, const CoarseSolver & coarse_solver,
                    double tau = 1.75) {
	if (hier.depth() == 1) {
		coarse_solver(b, x);
	} else {
		relaxed_wcycle(0, b, x, hier, coarse_solver, tau);
	}
}

}

#endif
