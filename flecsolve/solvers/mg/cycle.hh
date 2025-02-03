#ifndef FLECSOLVE_SOLVERS_MG_CYCLE_HH
#define FLECSOLVE_SOLVERS_MG_CYCLE_HH

#include <cstddef>
#include <istream>

namespace flecsolve::mg {

enum class cycle_type { v, relaxed_w, relaxed_kappa, kappa_k };

inline std::istream & operator>>(std::istream & in, cycle_type & ctype) {
	std::string tok;
	in >> tok;

	if (tok == "v")
		ctype = cycle_type::v;
	else if (tok == "relaxed-w")
		ctype = cycle_type::relaxed_w;
	else if (tok == "relaxed-kappa")
		ctype = cycle_type::relaxed_kappa;
	else if (tok == "kappa-k")
		ctype = cycle_type::kappa_k;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

inline std::ostream & operator<<(std::ostream & os, const cycle_type & ctype) {
	if (ctype == cycle_type::v)
		os << "v";
	else if (ctype == cycle_type::relaxed_w)
		os << "relaxed-w";
	else if (ctype == cycle_type::relaxed_kappa)
		os << "relaxed-kappa";
	else if (ctype == cycle_type::kappa_k)
	    os << "kappa-k";

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

template<class D, class R, class Hier, class CoarseSolver>
void relaxed_kappa_cycle(std::size_t lvl, const D & b, R & x,
                         Hier & hier, const CoarseSolver & coarse_solver,
                         double tau, int kappa) {
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

		if (kappa > 1) {
			c.set_scalar(0.);
			relaxed_kappa_cycle(lvl+1, coarse_b, c, hier, coarse_solver, tau, kappa);
			Ac(c, v);
			btilde.axpy(-tau, v, coarse_b);
			d.set_scalar(0.);
			relaxed_kappa_cycle(lvl+1, btilde, d, hier, coarse_solver, tau, kappa - 1);
			coarse_x.linear_sum(tau, c, tau, d);
		} else {
			relaxed_kappa_cycle(lvl+1, coarse_b, coarse_x, hier, coarse_solver, tau, kappa);
		}
	}

	auto & correction = level.correction();
	clevel.P()(coarse_x, correction);
	x.add(x, correction);
	level.postsmoother()(b, x);
}


template<class D, class R, class Hier, class CoarseSolver>
void relaxed_kappa_cycle(const D & b, R & x,
                         Hier & hier, const CoarseSolver & coarse_solver,
                         double tau = 1.75, int kappa = 1) {
	if (hier.depth() == 1) {
		coarse_solver(b, x);
	} else {
		relaxed_kappa_cycle(0, b, x, hier, coarse_solver, tau, kappa);
	}
}

template<class D, class R, class Hier, class CoarseSolver>
void kappa_kcycle(std::size_t lvl, const D & b, R & x,
                  Hier & hier, const CoarseSolver & coarse_solver,
                  int kappa, float ktol) {
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
        auto & w = clevel.w();

		if (kappa > 1) {
			c.set_scalar(0.);
			kappa_kcycle(lvl+1, coarse_b, c, hier, coarse_solver, kappa, ktol);
			Ac(c, v);
            auto rho1 = c.dot(v).get();
            auto alpha1 = c.dot(coarse_b).get();
            auto tau1 = alpha1 / rho1;
			btilde.axpy(-tau1, v, coarse_b);
            if (btilde.l2norm().get() < ktol * coarse_b.l2norm().get()) {
	            coarse_x.axpy(tau1, c, coarse_x);
            } else {
	            d.set_scalar(0.);
	            kappa_kcycle(lvl+1, btilde, d, hier, coarse_solver, kappa - 1, ktol);
	            Ac(d, w);
	            auto gamma = d.dot(v).get();
	            auto beta = d.dot(w).get();
	            auto alpha2 = d.dot(btilde).get();
	            auto rho2 = beta - ((gamma*gamma) / rho1);
	            auto tau2 = tau1 - (gamma*alpha2) / (rho1*rho2);
	            auto tau3 = alpha2 / rho2;
	            coarse_x.linear_sum(tau2, c, tau3, d);
            }
		} else {
			coarse_x.set_scalar(0.0);
			kappa_kcycle(lvl+1, coarse_b, coarse_x, hier, coarse_solver, kappa, ktol);
		}
	}

	auto & correction = level.correction();
	clevel.P()(coarse_x, correction);
	x.add(x, correction);
	level.postsmoother()(b, x);
}


template<class D, class R, class Hier, class CoarseSolver>
void kappa_kcycle(const D & b, R & x,
                  Hier & hier, const CoarseSolver & coarse_solver,
                  int kappa = 1, float ktol = 0.25) {
	if (hier.depth() == 1) {
		coarse_solver(b, x);
	} else {
        kappa_kcycle(0, b, x, hier, coarse_solver, kappa, ktol);
	}
}

}

#endif
