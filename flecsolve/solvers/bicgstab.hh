/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
#ifndef FLECSI_LINALG_OP_BICGSTAB_H
#define FLECSI_LINALG_OP_BICGSTAB_H

#include <flecsi/flog.hh>
#include <flecsi/execution.hh>

#include "solver_settings.hh"
#include "krylov_parameters.hh"

namespace flecsolve::op {
template<class Params>
struct bicgstab : base<Params,
                       typename Params::input_var_t,
                       typename Params::output_var_t>
{
	using base_t = base<Params, typename Params::input_var_t, typename Params::output_var_t>;
	using base_t::params;
	using real = typename Params::real;

	bicgstab(Params p) : base_t(std::move(p)) {}

	const auto & get_operator() const { return params.A(); }

	template<class DomainVec, class RangeVec>
	solve_info apply(const RangeVec & b,
	                 DomainVec & x) const {
		solve_info info;

		const auto & A = params.A();
		const auto & P = params.P();
		auto & user_diagnostic = params.ops.diagnostic;
		const auto & settings = params.settings;

		auto & [res, r_tilde, p, v, p_hat, s, s_hat, t] = params.work;

		real b_norm = b.l2norm().get();

		// if the rhs is zero try to converge to the relative convergence
		if (b_norm == 0.) {
			b_norm = 1.;
		}

		const real terminate_tol = settings.rtol * b_norm;

		info.rhs_norm = b_norm;

		if (settings.use_zero_guess) {
			info.sol_norm_initial = 0;
			res.copy(b);
			x.set_scalar(0.);
		}
		else {
			info.sol_norm_initial = x.l2norm().get();
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
		for (int iter = 0; iter < settings.maxiter; iter++) {
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
			}
			else {
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
				info.iters = iter + 1;
				break;
			}

			if (res_norm < terminate_tol) {
				info.status = solve_info::stop_reason::converged_rtol;
				info.iters = iter + 1;
				break;
			}

			if (omega == 0.0) {
				info.iters = iter + 1;
				info.status = solve_info::stop_reason::diverged_breakdown;
				break;
			}

			rho[0] = rho[1];
		}

		info.res_norm_final = res_norm;
		info.sol_norm_final = x.l2norm().get();
		if (info.iters == 0)
			info.status = solve_info::stop_reason::diverged_iters;

		return info;
	}
};
template<class P>
bicgstab(P)->bicgstab<P>;
}

namespace flecsolve::bicgstab {

static constexpr std::size_t nwork = 8;

struct settings : solver_settings {};
struct options : solver_options {
	using settings_type = settings;
	options(const char * pre) : solver_options(pre) {}
};

static inline work_factory<nwork> make_work;

template<class Workspace>
struct solver : krylov_solver<op::bicgstab, settings, Workspace> {
	using base_t = krylov_solver<op::bicgstab, settings, Workspace>;

	template<class W>
	solver(const settings & s, W && w) :
		base_t{s, std::forward<W>(w)} {}
};
template<class W>
solver(const settings &, W &&) -> solver<std::decay_t<W>>;

}
#endif
