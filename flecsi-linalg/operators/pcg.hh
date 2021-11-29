#pragma once

#include <iostream>

#include <flecsi/flog.hh>

namespace flecsi::linalg::pcg {

template <class Op, class Vec>
struct settings {
	using real_t = typename Vec::real_t;

	settings(std::array<Vec, 4> temps,
	         Op precond,
	         int maxiter=100, real_t rtol = 1e-9) :
		maxiter(maxiter), rtol(rtol), precond(std::move(precond)),
		temp{std::move(temps)} {}

	int maxiter;
	real_t rtol;
	Op precond;
	std::array<Vec, 4> temp;
};


template <class Op, class Vec>
struct flecsi_settings : settings<Op, Vec>
{
	using field_def = typename Vec::data_t::field_definition;
	using real_t = typename settings<Op, Vec>::real_t;
	using topo_slot_t = typename Vec::data_t::topo_slot_t;
	static inline std::array<const field_def, 4> defs;

	flecsi_settings(topo_slot_t & topo,
	                Op precond, int maxiter=100, real_t rtol = 1e-9) :
		settings<Op, Vec>({{
					{{defs[0], topo}}, {{defs[1], topo}},
					{{defs[2], topo}}, {{defs[3], topo}}
				}},
			std::move(precond), maxiter, rtol) {}
};


template<class Settings>
class solver
{
public:
	using real_t = typename Settings::real_t;

	solver(Settings params) : params{std::move(params)} {}

	template<class Op, class DomainVec, class RangeVec>
	void apply(const Op & A, const RangeVec & b, DomainVec & x)
	{
		auto & [r, z, p, w] = params.temp;
		auto & P = params.precond;
		const real_t b_norm = b.l2norm().get();

		if (b_norm == 0.0) return;

		const real_t terminate_tol = params.rtol * b_norm;


		flog(info) << "PCG: initial l2 norm of solution: " << x.l2norm().get() << std::endl;
		flog(info) << "PCG: initial l2 norm of rhs:      " << b_norm << std::endl;

		// compute initial residual
		A.residual(b, x, r);

		real_t current_res = r.l2norm().get();

		if (current_res < terminate_tol) return;

		P.apply(r, z);

		std::array<real_t, 2> rho{2.0, 0.0};
		rho[1] = z.inner_prod(r).get();
		rho[0] = rho[1];

		p.copy(z);
		for (auto iter = 0; iter < params.maxiter; iter++) {
			real_t beta = 1.0;

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
			flog(info) << "||r_" << iter << "|| " << current_res << std::endl;
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
