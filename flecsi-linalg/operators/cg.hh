#pragma once

#include <flecsi/flog.hh>

#include "shell.hh"
#include "solver_settings.hh"

namespace flecsi::linalg::cg {

static constexpr std::size_t nwork = 4;

template <class Op> using settings = solver_settings<Op>;

inline auto default_settings() {
	return settings<decltype(op::I)>{100, 1e-9, 1e-9, op::I};
}


template <std::size_t Version = 0>
using topo_work = topo_work_base<nwork, Version>;


template<class Settings, class Workspace>
struct solver : solver_interface<Settings, Workspace, solver>
{
	using iface = solver_interface<Settings, Workspace, solver>;
	using iface::work;
	using iface::settings;
	using iface::apply;

	template<class S, class V>
	solver(S && params, V && workspace) :
		iface{std::forward<S>(params),std::forward<V>(workspace)} {}

	template<class Op, class DomainVec, class RangeVec, class F>
	void apply(const Op & A, const RangeVec & b, DomainVec & x, F && callback)
	{
		using scalar = typename DomainVec::scalar;
		using real = typename DomainVec::real;

		auto & [r, z, p, w] = work;
		auto & P = settings.precond;
		real b_norm = b.l2norm().get();

		if (b_norm == 0.0) b_norm = 1.0;

		const real terminate_tol = settings.rtol * b_norm;


		flog(info) << "CG: initial l2 norm of solution: " << x.l2norm().get() << std::endl;
		flog(info) << "CG: initial l2 norm of rhs:      " << b_norm << std::endl;

		// compute initial residual
		A.residual(b, x, r);

		real current_res = r.l2norm().get();

		if (current_res < terminate_tol) return;

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
				flog(error) << "PCG: negative curvature encountered!" << std::endl;
			}

			alpha = rho[1] / alpha;

			x.axpy(alpha, p, x);  // x = x + alpha * p
			r.axpy(-alpha, w, r); // r = r - alpha * w

			current_res = r.l2norm().get();
			flog(info) << "CG: ||r_" << iter+1 << "|| " << current_res << std::endl;
			iface::invoke(std::forward<F>(callback), x, current_res);

			if (current_res < terminate_tol) break;

			P.apply(r, z);

			rho[0] = rho[1];
			rho[1] = r.dot(z).get();

			beta = rho[1] / rho[0];
			p.axpy(beta, p, z);
		}
	}
};
template<class S, class V> solver(S&&,V&&)->solver<S,V>;

}
