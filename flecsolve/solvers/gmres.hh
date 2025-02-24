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
#ifndef FLECSI_LINALG_OP_GMRES_H
#define FLECSI_LINALG_OP_GMRES_H

#include <cmath>

#include <flecsi/flog.hh>
#include <flecsi/util/array_ref.hh>
#include <flecsi/execution.hh>
#include <functional>

#include "solver_settings.hh"
#include "krylov_operator.hh"

namespace flecsolve::gmres {

enum class precond_side { left, right };

inline std::istream & operator>>(std::istream & in, precond_side & s) {
	std::string tok;
	in >> tok;

	if (tok == "left")
		s = precond_side::left;
	else if (tok == "right")
		s = precond_side::right;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}
inline std::ostream & operator<<(std::ostream & os, const precond_side & s) {
	if (s == precond_side::left)
		os << "left";
	else if (s == precond_side::right)
		os << "right";
	return os;
}

static constexpr int krylov_dim_bound = 100;
static constexpr std::size_t nwork = (krylov_dim_bound + 1) + 3;

struct settings : solver_settings {
	int max_krylov_dim;
	precond_side pre_side;
	bool restart;

	void validate() {
		if (max_krylov_dim < 0)
			max_krylov_dim = maxiter;
		flog_assert(max_krylov_dim <= krylov_dim_bound,
		            "GMRES: max_krylov_dim is larger than bound");
		if (!restart) {
			flog_assert(maxiter <= max_krylov_dim,
			            "GMRES: maxiters must be less than or equal to "
			            "max_krylov_dim when not using restart");
		}
	}
};
struct options : solver_options {
	using settings_type = settings;
	using base_t = solver_options;
	options(const char * pre) : solver_options(pre) {}

	auto operator()(settings_type & s) {
		auto desc = solver_options::operator()(s);
		// clang-format off
		desc.add_options()
			(label("max-krylov-dim").c_str(), po::value<int>(&s.max_krylov_dim)->default_value(-1), "maximum krylov dimension")
			(label("pre-side").c_str(),
			 po::value<precond_side>(&s.pre_side)->default_value(precond_side::right), "preconditioner side")
			(label("restart").c_str(), po::value<bool>(&s.restart)->default_value(false), "should restart");
		// clang-format on
		return desc;
	}
};

template<std::size_t Version = 0>
using topo_work = topo_work_base<nwork, Version>;

template<class Workspace>
struct solver : krylov_interface<Workspace> {
	using settings_type = settings;
	using iface = krylov_interface<Workspace>;
	using real = typename iface::real;
	using iface::work;

	template<class V>
	solver(const settings & params, V && workspace)
		: iface{std::forward<V>(workspace)}, params(params) {
		reset();
	}

	void reset(const settings & params) {
		this->params = params;
		reset();
	}

	void reset() {
		this->params.validate();

		int max_dim = std::min(params.max_krylov_dim, params.maxiter);

		hessenberg_data =
			std::make_unique<real[]>((max_dim + 1) * (max_dim + 1));
		hmat = std::make_unique<hessenberg_mat>(
			hessenberg_data.get(),
			std::array<std::size_t, 2>{static_cast<std::size_t>(max_dim) + 1,
		                               static_cast<std::size_t>(max_dim) + 1});
		auto & hessenberg = *hmat;
		for (int j = 0; j < max_dim + 1; j++) {
			for (int i = 0; i < max_dim + 1; i++) {
				hessenberg(i, j) = 0.0;
			}
		}
		cosvec.resize(max_dim + 1, 0.0);
		sinvec.resize(max_dim + 1, 0.0);
		dwvec.resize(max_dim + 1, 0.0);
		dyvec.resize(max_dim + 1, 0.0);
	}

	template<class Op, class DomainVec, class RangeVec, class Pre, class F>
	solve_info apply(const Op & A,
	                 const RangeVec & b,
	                 DomainVec & x,
	                 Pre & P,
	                 F && user_diagnostic) {
		solve_info info;
		auto & hessenberg = *hmat;
		auto basis = get_basis();

		std::size_t wrk = 0;
		auto & res = work[wrk++];
		auto & z = work[wrk++];
		auto & v = work[wrk++];
		flog_assert(wrk == (nwork - krylov_dim_bound - 1),
		            "GMRES: incorrect number of work vectors");

		auto b_norm = b.l2norm().get();

		info.rhs_norm = b_norm;

		// if rhs is zero try to converge to relative convergence
		if (b_norm < std::numeric_limits<real>::epsilon())
			b_norm = 1.0;

		const real terminate_tol = params.rtol * b_norm;

		if (params.use_zero_guess)
			x.set_scalar(0.);

		if (params.pre_side == precond_side::left) {
			if (params.use_zero_guess)
				basis[0].copy(b);
			else
				A.residual(b, x, basis[0]);
			P.apply(basis[0], res);
		}
		else {
			if (params.use_zero_guess)
				res.copy(b);
			else
				A.residual(b, x, res);
		}

		const real beta = res.l2norm().get();
		info.res_norm_initial = beta;

		if (beta < terminate_tol) {
			info.res_norm_final = beta;
			info.status = solve_info::stop_reason::converged_rtol;
			return info;
		}

		res.scale(1.0 / beta);
		basis[0].copy(res);

		// 'w*e_1' is the rhs for the least squares problem
		dwvec[0] = beta;
		auto v_norm = beta;

		int k = 0;
		trace.skip();
		auto g = std::make_unique<flecsi::exec::trace::guard>(trace);
		for (int iter = 0; iter < params.maxiter; iter++) {
			if (params.pre_side == precond_side::right) {
				P.apply(basis[k], z);
				A.apply(z, v);
			}
			else {
				A.apply(basis[k], z);
				P.apply(z, v);
			}

			// orthogonalize to previous vectors and add new colum to Hessenberg
			// matrix
			orthogonalize(v, k + 1);

			v_norm = hessenberg(k + 1, k);
			if (v_norm != 0.0) {
				v.scale(1.0 / v_norm);
			}

			// update basis with new orthonormal vector
			basis[k + 1].copy(v);

			// apply all previous Givens rotations to kth column of Hessenberg
			// matrix
			for (int i = 0; i < k; i++) {
				apply_givens_rotation(i, k);
			}

			if (v_norm != 0.0) {
				// compute and store the Givens rotation that zeroes out
				// the subdiagonal for the current column
				compute_givens_rotation(k);
				// zero out the subdiagonal
				apply_givens_rotation(k, k);
				hessenberg(k + 1, k) =
					0.0; // explicitly set subdiag to zero to prevent round-off

				// explicitly apply the newly computed
				// Givens rotations to the rhs vector
				auto x = dwvec[k];
				auto c = cosvec[k];
				auto s = sinvec[k];
#if 0
				dwvec[k]     = c * x;
				dwvec[k + 1] = s * x;
#else
				dwvec[k] = c * x;
				dwvec[k + 1] = -s * x;
#endif
			}

			v_norm = std::fabs(dwvec[k + 1]);
			++k;

			if (user_diagnostic(x, v_norm)) {
				info.status = solve_info::stop_reason::converged_user;
				info.iters = iter + 1;
				break;
			}

			if (v_norm < terminate_tol) {
				info.status = solve_info::stop_reason::converged_rtol;
				info.iters = iter + 1;
				break;
			}

			if (k == params.max_krylov_dim && iter != params.maxiter - 1) {
				back_solve(k - 1);
				correct(k - 1, P, z, v, x);

				if (params.pre_side == precond_side::left) {
					A.residual(b, x, basis[0]);
					P.apply(basis[0], res);
				}
				else
					A.residual(b, x, res);
				const real betar = res.l2norm().get();
				res.scale(1.0 / betar);
				basis[0].copy(res);
				dwvec[0] = betar;

				++info.restarts;
				k = 0;
				g.reset();
				g.reset(new flecsi::exec::trace::guard(trace));
			}
		}
		g.reset();

		if (k > 0) {
			back_solve(k - 1);

			// update current approximation with correction
			correct(k - 1, P, z, v, x);
		}

		info.res_norm_final = v_norm;
		info.sol_norm_final = x.l2norm().get();
		if (info.iters == 0)
			info.status = solve_info::stop_reason::diverged_iters;

		return info;
	}

protected:
	template<class Op, class W, class T>
	void correct(int nr, Op & P, W & z, W & v, T & x) {
		auto basis = get_basis();
		if (params.pre_side == precond_side::right) {
			z.set_scalar(0.0);

			for (int i = 0; i <= nr; i++) {
				z.axpy(dyvec[i], basis[i], z);
			}

			P.apply(z, v);
			x.axpy(1.0, v, x);
		}
		else {
			for (int i = 0; i <= nr; i++) {
				x.axpy(dyvec[i], basis[i], x);
			}
		}
	}

	template<class T>
	void orthogonalize(T & v, int k) {
		auto & hessenberg = *hmat;
		auto basis = get_basis();
		// modified Gram-Schmidt
		for (int j = 0; j < k; j++) {
			const double h_jk = v.dot(basis[j]).get();
			v.axpy(-h_jk, basis[j], v);
			hessenberg(j, k - 1) = h_jk;
		}

		// h_{k+1, k}
		const auto v_norm = v.l2norm().get();
		hessenberg(k, k - 1) = v_norm; // adjusting for zero starting index
	}

	void apply_givens_rotation(int i, int k) {
		auto & hessenberg = *hmat;

		auto x = hessenberg(i, k);
		auto y = hessenberg(i + 1, k);
		auto c = cosvec[i];
		auto s = sinvec[i];

#if 0
		hessenberg( i, k )     = c * x - s * y;
		hessenberg( i + 1, k ) = s * x + c * y;
#else
		hessenberg(i, k) = c * x + s * y;
		hessenberg(i + 1, k) = -s * x + c * y;
#endif
	}

	void compute_givens_rotation(int k) {
		// computes the Givens rotation required to zero out
		// the subdiagonal on column k of the Hessenberg matrix

		// The implementation here follows Algorithm 1 in
		// "On Computing Givens rotations reliably and efficiently"
		// by D. Bindel, J. Demmel, W. Kahan, O. Marques
		// UT-CS-00-449, October 2000.

		auto & hessenberg = *hmat;

		auto f = hessenberg(k, k);
		auto g = hessenberg(k + 1, k);

		real c, s;
		if (g == 0.0) {
			c = 1.0;
			s = 0.0;
		}
		else if (f == 0.0) {
			c = 0.0;
			s = (g < 0.0) ? -1.0 : 1.0;
		}
		else {
			real r;
			r = std::sqrt(f * f + g * g);
			r = 1.0 / r;
			c = std::fabs(f) * r;
			s = std::copysign(g * r, f);
		}

		cosvec[k] = c;
		sinvec[k] = s;
	}

	void back_solve(int nr) {
		auto & hessenberg = *hmat;
		// lower corner
		dyvec[nr] = dwvec[nr] / hessenberg(nr, nr);

		// back solve
		for (int k = nr - 1; k >= 0; k--) {
			dyvec[k] = dwvec[k];

			for (int i = k + 1; i <= nr; i++) {
				dyvec[k] -= hessenberg(k, i) * dyvec[i];
			}

			dyvec[k] = dyvec[k] / hessenberg(k, k);
		}
	}

	flecsi::util::span<typename iface::workvec_t> get_basis() {
		return {work.data() + (nwork - krylov_dim_bound - 1), cosvec.size()};
	}

protected:
	std::unique_ptr<real[]> hessenberg_data;
	using hessenberg_mat = flecsi::util::mdcolex<real, 2>;
	std::unique_ptr<hessenberg_mat> hmat;
	std::vector<real> sinvec, cosvec;
	std::vector<real> dwvec, dyvec;
	settings params;
	flecsi::exec::trace trace;
};
template<class V>
solver(const settings &, V &&) -> solver<V>;

}

namespace flecsolve {
template<>
struct traits<gmres::settings> {
	template<class W>
	using solver_type = gmres::solver<W>;
};
}
#endif
