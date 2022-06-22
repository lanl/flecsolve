#ifndef FLECSI_LINALG_OP_CG_H
#define FLECSI_LINALG_OP_CG_H

#include <flecsi/flog.hh>
#include <flecsi/execution.hh>

#include "krylov_interface.hh"
#include "shell.hh"
#include "solver_settings.hh"

namespace flecsolve::cg {

static constexpr std::size_t nwork = 4;

using settings = solver_settings;

template<std::size_t Version = 0>
using topo_work = topo_work_base<nwork, Version>;

template<class Workspace>
struct solver : krylov_interface<Workspace, solver> {
	using settings_type = settings;
	using iface = krylov_interface<Workspace, solver>;
	using iface::work;

	template<class V>
	solver(const settings & params, V && workspace)
		: iface{std::forward<V>(workspace)}, params(params) {}

	void reset(const settings & params) { this->params = params; }

	template<class Op,
	         class DomainVec,
	         class RangeVec,
	         class Precond,
	         class Diag>
	solve_info apply(const Op & A,
	                 const RangeVec & b,
	                 DomainVec & x,
	                 Precond & P,
	                 Diag && user_diagnostic) {
		solve_info info;
		using scalar = typename DomainVec::scalar;
		using real = typename DomainVec::real;

		auto & [r, z, p, w] = work;
		real b_norm = b.l2norm().get();

		if (b_norm == 0.0)
			b_norm = 1.0;

		const real terminate_tol = params.rtol * b_norm;

		info.rhs_norm = b_norm;

		// compute initial residual
		if (params.use_zero_guess) {
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
		trace.skip();
		for (auto iter = 0; iter < params.maxiter; iter++) {
			auto g = trace.make_guard();
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

protected:
	settings params;
	flecsi::exec::trace trace;
};

template<class V>
solver(const settings &, V &&) -> solver<V>;

} // namespace flecsolve::cg

namespace flecsolve {
template<>
struct traits<cg::settings> {
	template<class W>
	using solver_type = cg::solver<W>;
};
} // namespace flecsolve
#endif
