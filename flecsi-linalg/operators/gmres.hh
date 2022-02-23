#pragma once

#include <cmath>

#include <flecsi/flog.hh>
#include <flecsi/util/array_ref.hh>

#include "solver_settings.hh"
#include "shell_operator.hh"

namespace flecsi::linalg::gmres {

enum class precond_side { left, right };

static constexpr std::size_t krylov_dim_bound = 100;
static constexpr std::size_t nwork = (krylov_dim_bound+1) + 3;

template <class Op, class Vec>
struct settings : solver_settings<Op, Vec, nwork>
{
	using base_t = solver_settings<Op, Vec, nwork>;
	settings(Op && precond, std::array<Vec, nwork> workvecs, int maxiter=100, double rtol=1e-9) :
		base_t{maxiter, rtol, 0.0, std::forward<Op>(precond), std::move(workvecs)},
		max_krylov_dim(100), pre_side{precond_side::right} {
		flog_assert(max_krylov_dim <= krylov_dim_bound, "GMRES: max_krylov_dim is larger than bound");
	}

	int max_krylov_dim;
	precond_side pre_side;
};


template <class Op, class Vec>
auto topo_settings(Vec & rhs,
                   Op && precond, int maxiter=100, double rtol=1e-9) {
	return settings<Op, Vec>(std::forward<Op>(precond),
	                         topo_solver_state<Vec, nwork>::get_work(rhs),
	                         maxiter, rtol);
}

template <class Vec>
auto topo_settings(Vec & rhs,
                   int maxiter=100, double rtol=1e-9) {
	shell_operator P{[](const auto & x, auto & y) { y.copy(x); }};
	return settings<decltype(P), Vec>(std::move(P),
	                                  topo_solver_state<Vec, nwork>::get_work(rhs),
	                                  maxiter, rtol);
}

template<class Settings>
struct solver {
	using real = typename Settings::real;

	template<class S>
	solver(S && params) : settings(std::forward<S>(params)) {
		init();
	}

	void init() {
		std::size_t max_dim = std::min(settings.max_krylov_dim, settings.maxiter);
		hessenberg_data = std::make_unique<real[]>((max_dim+1)*(max_dim+1));
		hmat = std::make_unique<hessenberg_mat>(hessenberg_data.get(),
		                                        std::array<std::size_t, 2>{max_dim + 1, max_dim + 1});
		auto & hessenberg = *hmat;
		for (int j = 0; j < max_dim + 1; j++) {
			for (int i = 0; i < max_dim + 1; i++) {
				hessenberg(i,j) = 0.0;
			}
		}
		cosvec.resize(max_dim + 1, 0.0);
		sinvec.resize(max_dim + 1, 0.0);
		dwvec.resize(max_dim + 1, 0.0);
		dyvec.resize(max_dim + 1, 0.0);

		basis = util::span(settings.work.data() + (nwork - krylov_dim_bound - 1), max_dim+1);
	}

	template<class Op, class DomainVec, class RangeVec>
	void apply(const Op & A, const RangeVec & b, DomainVec & x)
	{
		apply(A, b, x, nullptr);
	}

	template<class Op, class DomainVec, class RangeVec, class F>
	void apply(const Op & A, const RangeVec & b, DomainVec & x, F && callback) {
		auto & P = settings.precond;
		auto & hessenberg = *hmat;

		std::size_t wrk = 0;
		auto & res = settings.work[wrk++];
		auto & z = settings.work[wrk++];
		auto & v = settings.work[wrk++];
		flog_assert(wrk == (nwork - krylov_dim_bound - 1), "GMRES: incorrect number of work vectors");

		using scalar = typename DomainVec::scalar;

		auto b_norm = b.l2norm().get();

		// if rhs is zero try to converge to relative convergence
		if (b_norm < std::numeric_limits<real>::epsilon())
			b_norm = 1.0;

		const real terminate_tol = settings.rtol * b_norm;

		A.residual(b, x, res);

		nr = -1;

		const real beta = res.l2norm().get();
		flog(info) << "gmres: initial residual " << beta << std::endl;

		if (beta < terminate_tol) return;

		res.scale(1.0 / beta);
		basis[0].copy(res);

		// 'w*e_1' is the rhs for the least squares problem
		dwvec[0] = beta;
		auto v_norm = beta;

		for (int k = 0; (k < settings.maxiter) and (v_norm > terminate_tol); k++) {
			if (settings.pre_side == precond_side::right)
				P.apply(basis[k], z);
			else
				z.copy(basis[k]);

			// construct krylov vector
			A.apply(z, v);

			// orthogonalize to previous vectors and add new colum to Hessenberg matrix
			orthogonalize(v, k+1);

			v_norm = hessenberg(k+1, k);
			if (v_norm != 0.0) {
				v.scale( 1.0 / v_norm);
			}

			// update basis with new orthonormal vector
			basis[k+1].copy(v);

			// apply all previous Givens rotations to kth column of Hessenberg matrix
			for (int i = 0; i < k; i++) {
				apply_givens_rotation(i, k);
			}

			if (v_norm != 0.0) {
				// compute and store the Givens rotation that zeroes out
				// the subdiagonal for the current column
				compute_givens_rotation( k );
				// zero out the subdiagonal
				apply_givens_rotation( k, k );
				hessenberg( k + 1, k ) = 0.0; // explicitly set subdiag to zero to prevent round-off

				// explicitly apply the newly computed
				// Givens rotations to the rhs vector
				auto x = dwvec[k];
				auto y = dwvec[k + 1];
				auto c = cosvec[k];
				auto s = sinvec[k];
#if 0
				dwvec[k]     = c * x - s * y;
				dwvec[k + 1] = s * x + c * y;
#else
				dwvec[k]     = c * x + s * y;
				dwvec[k + 1] = -s * x + c * y;
#endif
			}

			v_norm = std::fabs(dwvec[k+1]);

			flog(info) << "|r_" << k + 1 << "| " << v_norm << std::endl;
			if constexpr (!std::is_null_pointer_v<F>) callback(x, v_norm);

			++nr;
		}

		back_solve();

		// update current approximation with correction
		if (settings.pre_side == precond_side::right) {
			z.set_scalar(0.0);

			for (int i = 0; i <= nr; i++) {
				z.axpy(dyvec[i], basis[i], z);
			}

			P.apply(z, v);
			x.axpy(1.0, v, x);
		} else {
			for (int i = 0; i <= nr; i++) {
				x.axpy(dyvec[i], basis[i], x);
			}
		}
	}


	void orthogonalize(typename Settings::vec & v, int k) {
		auto & hessenberg = *hmat;
		// modified Gram-Schmidt
		for (int j = 0; j < k; j++) {
			const double h_jk = v.inner_prod(basis[j]).get();
			v.axpy(-h_jk, basis[j], v);
			hessenberg(j, k-1) = h_jk;
		}

		// h_{k+1, k}
		const auto v_norm = v.l2norm().get();
		hessenberg(k, k-1) = v_norm; // adjusting for zero starting index
	}

	void apply_givens_rotation(int i, int k) {
		auto & hessenberg = *hmat;

		auto x = hessenberg( i, k );
		auto y = hessenberg( i + 1, k );
		auto c = cosvec[i];
		auto s = sinvec[i];

#if 0
		hessenberg( i, k )     = c * x - s * y;
		hessenberg( i + 1, k ) = s * x + c * y;
#else
		hessenberg( i, k ) = c * x + s * y;
		hessenberg( i + 1, k ) = -s * x + c * y;
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
		auto g = hessenberg(k+1, k);

		real c, s;
		if ( g == 0.0 ) {
			c = 1.0;
			s = 0.0;
		} else if ( f == 0.0 ) {
			c = 0.0;
			s = ( g < 0.0 ) ? -1.0 : 1.0;
		} else {
			real r;
			r = std::sqrt( f * f + g * g );
			r = 1.0 / r;
			c = std::fabs( f ) * r;
			s = std::copysign( g * r, f );
		}

		cosvec[k] = c;
		sinvec[k] = s;
	}

	void back_solve() {
		auto & hessenberg = *hmat;
		// lower corner
		dyvec[nr] = dwvec[nr] / hessenberg(nr, nr);

		// back solve
		for (int k = nr - 1; k >= 0; k--) {
			dyvec[k] = dwvec[k];

			for (int i = k + 1; i <= nr; i++) {
				dyvec[k] -= hessenberg( k, i ) * dyvec[i];
			}

			dyvec[k] = dyvec[k] / hessenberg( k, k );
		}
	}


protected:
	Settings settings;
	std::unique_ptr<real[]> hessenberg_data;
	using hessenberg_mat = util::mdcolex<real, 2>;
	std::unique_ptr<hessenberg_mat> hmat;
	util::span<typename Settings::vec> basis;
	std::vector<real> sinvec, cosvec;
	std::vector<real> dwvec, dyvec;
	real nr;
};
template <class S> solver(S &&) -> solver<S>;

}
