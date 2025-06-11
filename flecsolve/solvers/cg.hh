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
#ifndef FLECSI_LINALG_OP_CG_H
#define FLECSI_LINALG_OP_CG_H

#include <flecsi/flog.hh>
#include <flecsi/execution.hh>

#include "solver_settings.hh"
#include "krylov_parameters.hh"

namespace flecsolve::op {

template<class Params>
struct cg : base<Params,
                 typename Params::input_var_t,
                 typename Params::output_var_t>
{
	using base_t = base<Params,
	                    typename Params::input_var_t,
	                    typename Params::output_var_t>;
	using base_t::params;
	using real = typename Params::real;
	using scalar = typename Params::scalar;

	cg(Params p) : base_t(std::move(p)) {}

	const auto & get_operator() const { return params.A(); }

	template<class DomainVec, class RangeVec>
	solve_info apply(const RangeVec & b, DomainVec & x) const {
		solve_info info;
		using scalar = typename DomainVec::scalar;
		using real = typename DomainVec::real;

		const auto & A = params.A();
		const auto & P = params.P();
		auto & user_diagnostic = params.ops.diagnostic;
		const auto & settings = params.settings;

		auto & [r, z, p, w] = params.work;
		real b_norm = b.l2norm().get();

		if (b_norm == 0.0)
			b_norm = 1.0;

		const real terminate_tol = settings.rtol * b_norm;

		info.rhs_norm = b_norm;

		// compute initial residual
		if (settings.use_zero_guess) {
			info.sol_norm_initial = 0;
			x.set_scalar(0.);
			r.copy(b);
		}
		else {
			info.sol_norm_initial = x.l2norm().get();
			A.residual(b, x, r);
		}

		real current_res = r.l2norm().get();

		if (current_res < terminate_tol) {
			info.res_norm_initial = current_res;
			info.res_norm_final = current_res;
			info.status = solve_info::stop_reason::converged_rtol;
			return info;
		}

		P.apply(r, z);

		std::array<scalar, 2> rho{2.0, 0.0};
		rho[1] = z.dot(r).get();
		rho[0] = rho[1];

		p.copy(z);
		for (auto iter = 0; iter < settings.maxiter; iter++) {
			scalar beta = 1.0;

			// w = Ap
			A.apply(p, w);

			// alpha = p'Ap
			auto alpha = w.dot(p).get();

			// sanity check, the curvature should be positive
			if (alpha <= 0.0) {
				flog(error)
					<< "PCG: negative curvature encountered!" << std::endl;
			}

			alpha = rho[1] / alpha;

			x.axpy(alpha, p, x); // x = x + alpha * p
			r.axpy(-alpha, w, r); // r = r - alpha * w

			current_res = r.l2norm().get();
			if (user_diagnostic(x, current_res)) {
				info.iters = iter + 1;
				info.status = solve_info::stop_reason::converged_user;
				break;
			}

			if (current_res < terminate_tol) {
				info.iters = iter + 1;
				info.status = solve_info::stop_reason::converged_rtol;
				break;
			}

			P.apply(r, z);

			rho[0] = rho[1];
			rho[1] = r.dot(z).get();

			beta = rho[1] / rho[0];
			p.axpy(beta, p, z);
		}

		info.res_norm_final = current_res;
		info.sol_norm_final = x.l2norm().get();
		if (info.iters == 0)
			info.status = solve_info::stop_reason::diverged_iters;

		return info;
	}
};
template<class P>
cg(P) -> cg<P>;


template<class Params>
struct fcg : base<Params,
                 typename Params::input_var_t,
                 typename Params::output_var_t>
{
	using base_t = base<Params,
	                    typename Params::input_var_t,
	                    typename Params::output_var_t>;
	using base_t::params;
	using real = typename Params::real;
	using scalar = typename Params::scalar;

	fcg(Params p) : base_t(std::move(p)) {}

	const auto & get_operator() const { return params.A(); }

	template<class DomainVec, class RangeVec>
	solve_info apply(const RangeVec & b, DomainVec & u) const {
		solve_info info;
		using scalar = typename DomainVec::scalar;
		using real = typename DomainVec::real;

		const auto & A = params.A();
		const auto & P = params.P();
		auto & user_diagnostic = params.ops.diagnostic;
		const auto & settings = params.settings;

		auto & [r, v, w, q, d] = params.work;
		real b_norm = b.l2norm().get();

		if (b_norm == 0.0)
			b_norm = 1.0;

		const real terminate_tol = settings.rtol * b_norm;

		info.rhs_norm = b_norm;

		// compute initial residual
		if (settings.use_zero_guess) {
			info.sol_norm_initial = 0;
			u.set_scalar(0.);
			r.copy(b);
		}
		else {
			info.sol_norm_initial = u.l2norm().get();
			A.residual(b, u, r);
		}

		real current_res = r.l2norm().get();

		if (current_res < terminate_tol) {
			info.res_norm_initial = current_res;
			info.res_norm_final = current_res;
			info.status = solve_info::stop_reason::converged_rtol;
			return info;
		}

		scalar rho = 0;
		for (auto iter = 0; iter < settings.maxiter; iter++) {
			P.apply(r, v);
			A.apply(v, w);

			scalar alpha = v.dot(r).get();
			scalar beta = v.dot(w).get();

			if (iter > 0) {
				scalar gamma = v.dot(q).get();
				d.axpy((-gamma)/rho, d, v);
				q.axpy((-gamma)/rho, q, w);
				rho = beta - (gamma*gamma) / rho;
			} else {
				d.copy(v);
				q.copy(w);
				rho = beta;
			}
			u.axpy(alpha / rho, d, u);
			r.axpy(-alpha / rho, q, r);

			current_res = r.l2norm().get();
			if (user_diagnostic(u, current_res)) {
				info.iters = iter + 1;
				info.status = solve_info::stop_reason::converged_user;
				break;
			}

			if (current_res < terminate_tol) {
				info.iters = iter + 1;
				info.status = solve_info::stop_reason::converged_rtol;
				break;
			}
		}

		info.res_norm_final = current_res;
		info.sol_norm_final = u.l2norm().get();
		if (info.iters == 0)
			info.status = solve_info::stop_reason::diverged_iters;

		return info;
	}
};
template<class P>
fcg(P) -> fcg<P>;

}

namespace flecsolve::cg {

static constexpr std::size_t nwork = 4;

using settings = solver_settings;
using options = solver_options;

static inline work_factory<nwork> make_work;

template<class Work>
struct solver : krylov_solver<op::cg, settings, Work> {
	using base_t = krylov_solver<op::cg, settings, Work>;

	template<class W>
	solver(const settings & set, W && w)
		: base_t{set, std::forward<W>(w)} {}
};
template<class W>
solver(const settings &, W &&) -> solver<std::decay_t<W>>;

}

namespace flecsolve::fcg {
static constexpr std::size_t nwork = 5;

using settings = solver_settings;
using options = solver_options;

static inline work_factory<nwork> make_work;

template<class Work>
struct solver : krylov_solver<op::fcg, settings, Work> {
	using base_t = krylov_solver<op::fcg, settings, Work>;

	template<class W>
	solver(const settings & set, W && w)
		: base_t{set, std::forward<W>(w)} {}
};
template<class W>
solver(const settings &, W &&) -> solver<std::decay_t<W>>;

}

#endif
