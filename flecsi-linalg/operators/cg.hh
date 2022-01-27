#pragma once

#include <flecsi/flog.hh>

#include "solver_settings.hh"

namespace flecsi::linalg::cg {

static constexpr std::size_t nwork = 4;

template <class Op, class Vec> using settings = solver_settings<Op, Vec, nwork>;

template <class Op, class Vec>
auto topo_settings(Vec & rhs,
                   Op precond, int maxiter=100, double rtol=1e-9) {
	return settings<Op,Vec>{maxiter, rtol, 0.0, std::move(precond),
		topo_solver_state<Vec, nwork>::get_work(rhs)};
}


template<class Settings>
class solver
{
public:
	using real = typename Settings::real;

	solver(Settings params) : params{std::move(params)} {}

	template<class Op, class DomainVec, class RangeVec>
	void apply(const Op & A, const RangeVec & b, DomainVec & x)
	{
		using scalar = typename DomainVec::scalar;

		auto & [r, z, p, w] = params.work;
		auto & P = params.precond;
		const real b_norm = b.l2norm().get();

		if (b_norm == 0.0) return;

		const real terminate_tol = params.rtol * b_norm;


		flog(info) << "CG: initial l2 norm of solution: " << x.l2norm().get() << std::endl;
		flog(info) << "CG: initial l2 norm of rhs:      " << b_norm << std::endl;

		// compute initial residual
		A.residual(b, x, r);

		real current_res = r.l2norm().get();

		if (current_res < terminate_tol) return;

		P.apply(r, z);

		std::array<scalar, 2> rho{2.0, 0.0};
		rho[1] = z.inner_prod(r).get();
		rho[0] = rho[1];

		p.copy(z);
		for (auto iter = 0; iter < params.maxiter; iter++) {
			scalar beta = 1.0;

			// w = Ap
			A.apply(p, w);

			// alpha = p'Ap
			auto alpha = w.inner_prod(p).get();

			// sanity check, the curvature should be positive
			if (alpha <= 0.0) {
				flog(error) << "PCG: negative curvature encountered!" << std::endl;
			}

			alpha = rho[1] / alpha;

			x.axpy(alpha, p, x);  // x = x + alpha * p
			r.axpy(-alpha, w, r); // r = r - alpha * w

			current_res = r.l2norm().get();
			flog(info) << "CG: ||r_" << iter+1 << "|| " << current_res << std::endl;
			if (current_res < terminate_tol) break;

			P.apply(r, z);

			rho[0] = rho[1];
			rho[1] = r.inner_prod(z).get();

			beta = rho[1] / rho[0];
			p.axpy(beta, p, z);
		}
	}

protected:
	Settings params;
};

}
