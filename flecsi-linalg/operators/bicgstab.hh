#pragma once

#include <flecsi/flog.hh>

#include "solver_settings.hh"
#include "shell.hh"

namespace flecsi::linalg::bicgstab {

static constexpr std::size_t nwork = 8;

template <class Op>
struct settings : solver_settings<Op>
{
	using base_t = solver_settings<Op>;
	template<class OP>
	settings(OP && precond, int maxiter, float rtol, bool use_zero_guess) :
		base_t{maxiter, rtol, 0.0, std::forward<OP>(precond)},
		use_zero_guess(use_zero_guess) {}

	bool use_zero_guess;
};

inline auto default_settings() {
	return settings<decltype(op::I)>(op::I, 100, 1e-9, false);
}


template <std::size_t Version = 0>
using topo_work = topo_work_base<nwork, Version>;


template <class Settings, class Workspace>
class solver
{
public:
	using real = typename Settings::real;

	template<class S, class W>
	solver(S && params, V && workspace) :
		settings{std::forward<S>(params)},
		work{std::forward<V>(workspace)} {}

	template<class Op, class DomainVec, class RangeVec>
	void apply(const Op & A, const RangeVec & b, DomainVec & x)
	{
		using scalar = typename DomainVec::scalar;
		int restarts = 0;

		auto & [res, r_tilde, p, v, p_hat, s, s_hat, t] = work;

		auto & P = settings.precond;

		real b_norm = b.l2norm().get();

		// if the rhs is zero try to converge to the relative convergence
		if (b_norm == 0.) {
			b_norm = 1.;
		}

		const real terminate_tol = settings.rtol * b_norm;

		flog(info) << "BiCGSTAB: initial l2 norm of solution: " << x.l2norm().get() << std::endl;
		flog(info) << "BiCGSTAB: initial l2 norm of rhs:      " << b_norm << std::endl;

		if (settings.use_zero_guess) {
			res.copy(b);
		} else {
			A.residual(b, x, res);
		}

		// compute current residual norm
		real res_norm = res.l2norm().get();
		real r_tilde_norm = res_norm;

		flog(info) << "BiCGSTAB: initial residual " << res_norm << std::endl;

		if (res_norm < terminate_tol) {
			flog(info) << "BiCGSTAB initial residual norm " << res_norm
			           << " is below convergence tolerance: " << terminate_tol << std::endl;
			return;
		}

		real alpha = 1.0;
		real beta = 0.0;
		real omega = 1.0;
		std::vector<real> rho{2, 1.0};

		// r_tilde is a non-zero initial direction chosen to be r
		// traditional choise is the initial residual
		r_tilde.copy(res);

		p.zero();
		v.zero();
		for (auto iter = 0; iter < settings.maxiter; iter++) {
			rho[1] = r_tilde.dot(res).get();

			real angle = std::sqrt(std::fabs(rho[1]));
			real eps = std::numeric_limits<real>::epsilon();

			if (angle < eps * r_tilde_norm) {
				// the method breaks down as the vectors are orthogonal to r0
				// attempt to restart with a new r0
				A.residual(b, x, res);
				r_tilde.copy(res);
				res_norm = res.l2norm().get();
				rho[1] = r_tilde_norm = res_norm;
				++restarts;
				break;
			}

			if (iter == 1) {
				p.copy(res);
			} else {
				beta = (rho[1] / rho[0]) * (alpha / omega);
				p.axpy(-omega, v, p);
				p.axpy(beta, p, res);
			}

			P.apply(p, p_hat);

			A.apply(p_hat, v);

			alpha = r_tilde.dot(v).get();
			flog_assert(alpha != 0., "BiCGSTAB: encountered alpha = 0");
			alpha = rho[1] / alpha;

			s.axpy(-alpha, v, res);

			const real s_norm = s.l2norm().get();

			if (s_norm < settings.rtol) {
				// early convergence
				x.axpy(alpha, p_hat, x);
				break;
			}

			P.apply(s, s_hat);

			A.apply(s_hat, t);

			real t_sqnorm = t.dot(t).get();
			real t_dot_s = t.dot(s).get();
			omega = (t_sqnorm == 0.0) ? 0.0 : t_dot_s / t_sqnorm;

			x.axpy(alpha, p_hat, x);
			x.axpy(omega, s_hat, x);

			res.axpy(-omega, t, s);

			res_norm = res.l2norm().get();

			flog(info) << "BiCGSTAB: ||r_" << (iter + 1) << "|| " << res_norm
			           << std::endl;

			if (res_norm < terminate_tol) break;

			if (omega == 0.0) {
				flog(error) << "BiCGSTAB: breakdown encountered, omega = 0" << std::endl;
				break;
			}

			rho[0] = rho[1];
		}

		flog(info) << "l2norm of solution: " << x.l2norm().get() << std::endl;
	}

	Settings settings;
	Workspace work;
};

}
