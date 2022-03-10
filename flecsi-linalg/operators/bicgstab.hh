#pragma once

#include <flecsi/flog.hh>

#include "solver_settings.hh"
#include "shell.hh"

namespace flecsi::linalg::bicgstab {

static constexpr std::size_t nwork = 8;

struct settings : solver_settings
{
	using base_t = solver_settings;
	settings(int maxiter, float rtol, bool use_zero_guess) :
		base_t{maxiter, rtol, 0.0},
		use_zero_guess(use_zero_guess) {}

	bool use_zero_guess;
};


inline auto default_settings() {
	return settings(100, 1e-9, false);
}

template <std::size_t Version = 0>
using topo_work = topo_work_base<nwork, Version>;


template <class Workspace>
struct solver : solver_interface<Workspace, solver>
{
	using iface = solver_interface<Workspace, solver>;
	using real = typename iface::real;
	using iface::work;

	template<class W>
	solver(const settings & params, W && workspace) :
		iface{std::forward<W>(workspace)}, params(params) {}

	void reset(const settings & params) {
		this->params = params;
	}

	template<class Op, class DomainVec, class RangeVec, class Pre, class F>
	solve_info apply(const Op & A, const RangeVec & b, DomainVec & x,
	                 Pre & P, F && user_diagnostic)
	{
		solve_info info;

		using scalar = typename DomainVec::scalar;

		auto & [res, r_tilde, p, v, p_hat, s, s_hat, t] = work;

		real b_norm = b.l2norm().get();

		// if the rhs is zero try to converge to the relative convergence
		if (b_norm == 0.) {
			b_norm = 1.;
		}

		const real terminate_tol = params.rtol * b_norm;

		info.sol_norm_initial = x.l2norm().get();
		info.rhs_norm = b_norm;

		if (params.use_zero_guess) {
			res.copy(b);
		} else {
			A.residual(b, x, res);
		}

		// compute current residual norm
		real res_norm = res.l2norm().get();
		real r_tilde_norm = res_norm;

		info.res_norm_initial = res_norm;

		if (res_norm < terminate_tol) {
			// initial residual below tolerance
			info.status = solve_info::stop_reason::converged_rtol;
			info.res_norm_initial = res_norm;
			info.res_norm_final = res_norm;
			return info;
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
		for (int iter = 0; iter < params.maxiter; iter++) {
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
				p.copy(res);
				++info.restarts;
				continue;
			}

			if (iter == 0) {
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

			if (s_norm < params.rtol) {
				// early convergence
				x.axpy(alpha, p_hat, x);

				info.iters = iter;
				info.status = solve_info::stop_reason::converged_rtol;
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

			if (user_diagnostic(x, res_norm)) {
				info.status = solve_info::stop_reason::converged_user;
				info.iters = iter+1;
				break;
			}

			if (res_norm < terminate_tol) {
				info.status = solve_info::stop_reason::converged_rtol;
				info.iters = iter+1;
				break;
			}

			if (omega == 0.0) {
				info.iters = iter+1;
				info.status = solve_info::stop_reason::diverged_breakdown;
				break;
			}

			rho[0] = rho[1];
		}

		info.res_norm_final = res_norm;
		info.sol_norm_final = x.l2norm().get();
		if (info.iters == 0) info.status = solve_info::stop_reason::diverged_iters;

		return info;
	}

protected:
	settings params;
};
template<class V> solver(const settings&,V&&)->solver<V>;

}
