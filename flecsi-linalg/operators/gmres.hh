#ifndef FLECSI_LINALG_OP_GMRES_H
#define FLECSI_LINALG_OP_GMRES_H

#include <cmath>

#include <flecsi/flog.hh>
#include <flecsi/util/array_ref.hh>

#include "solver_settings.hh"
#include "shell.hh"
#include "krylov_interface.hh"

namespace flecsi::linalg::gmres {

enum class precond_side { left, right };

static constexpr std::size_t krylov_dim_bound = 100;
static constexpr std::size_t nwork = (krylov_dim_bound+1) + 3;

struct settings : solver_settings
{
	using base_t = solver_settings;
	settings(int maxiter, float rtol, int restart) :
		base_t{maxiter, rtol, 0.0},
		max_krylov_dim(100), pre_side{precond_side::right}, restart(restart) {
		flog_assert(max_krylov_dim <= krylov_dim_bound, "GMRES: max_krylov_dim is larger than bound");
	}

	int max_krylov_dim;
	precond_side pre_side;
	int restart;
};

template <std::size_t Version = 0>
using topo_work = topo_work_base<nwork, Version>;


template<class Workspace>
struct solver : krylov_interface<Workspace, solver>
{
	using settings_type = settings;
	using iface = krylov_interface<Workspace, solver>;
	using real = typename iface::real;
	using iface::work;

	solver(solver<Workspace>&& o) noexcept :
		iface{std::move(o.work)},
		hessenberg_data(std::exchange(o.hessenberg_data, nullptr)),
		hmat(std::exchange(o.hmat, nullptr)),
		basis(work.data() + (nwork - krylov_dim_bound - 1), o.basis.size()),
		sinvec(std::move(o.sinvec)), cosvec(std::move(o.cosvec)),
		dwvec(std::move(o.dwvec)), dyvec(std::move(o.dyvec)), params(std::move(o.params))
	{}

	solver<Workspace> & operator=(solver<Workspace>&& o) noexcept {
		if (this != &o) {
			work = std::move(o.work);
			hessenberg_data = std::exchange(o.hessenberg_data, nullptr);
			hmat = std::exchange(o.hmat, nullptr);
			basis = util::span(work.data() + (nwork - krylov_dim_bound - 1), o.basis.size());
			sinvec = std::move(o.sinvec);
			cosvec = std::move(o.cosvec);
			dwvec = std::move(o.dwvec);
			dyvec = std::move(o.dyvec);
			params = std::move(o.params);
		}
		return *this;
	}

	template<class V>
	solver(const settings & params, V && workspace) :
		iface{std::forward<V>(workspace)}, params(params)
	{
		reset();
	}

	void reset(const settings & params) {
		this->params = params;
		reset();
	}

	void reset() {
		std::size_t max_dim = std::min(params.max_krylov_dim, params.maxiter);
		if (params.restart > 0)
			max_dim = std::min(max_dim, static_cast<std::size_t>(params.restart));

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

		basis = util::span(work.data() + (nwork - krylov_dim_bound - 1), max_dim+1);
	}

	template<class Op, class DomainVec, class RangeVec, class Pre, class F>
	solve_info apply(const Op & A, const RangeVec & b, DomainVec & x,
	                 Pre & P, F && user_diagnostic) {
		solve_info info;
		auto & hessenberg = *hmat;

		std::size_t wrk = 0;
		auto & res = work[wrk++];
		auto & z = work[wrk++];
		auto & v = work[wrk++];
		flog_assert(wrk == (nwork - krylov_dim_bound - 1), "GMRES: incorrect number of work vectors");

		using scalar = typename DomainVec::scalar;

		auto b_norm = b.l2norm().get();

		info.rhs_norm = b_norm;

		// if rhs is zero try to converge to relative convergence
		if (b_norm < std::numeric_limits<real>::epsilon())
			b_norm = 1.0;

		const real terminate_tol = params.rtol * b_norm;

		A.residual(b, x, res);

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
		for (int iter = 0; iter < params.maxiter; iter++) {
			if (params.pre_side == precond_side::right)
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
				auto c = cosvec[k];
				auto s = sinvec[k];
#if 0
				dwvec[k]     = c * x;
				dwvec[k + 1] = s * x;
#else
				dwvec[k]     = c * x;
				dwvec[k + 1] = -s * x;
#endif
			}

			v_norm = std::fabs(dwvec[k+1]);

			if (user_diagnostic(x, v_norm)) {
				info.status = solve_info::stop_reason::converged_user;
				info.iters = iter+1;
				break;
			}

			if (v_norm < terminate_tol) {
				info.status = solve_info::stop_reason::converged_rtol;
				info.iters = iter+1;
				break;
			}

			++k;
			if (k == params.restart and iter != params.maxiter-1) {
				back_solve(k-1);
				correct(k-1, P, z, v, x);

				A.residual(b, x, res);
				const real betar = res.l2norm().get();
				res.scale(1.0 / betar);
				basis[0].copy(res);
				dwvec[0] = betar;

				++info.restarts;
				k = 0;
			}
		}

		if (k > 0) {
			back_solve(k-1);

			// update current approximation with correction
			correct(k-1, P, z, v, x);
		}

		info.res_norm_final = v_norm;
		info.sol_norm_final = x.l2norm().get();
		if (info.iters == 0) info.status = solve_info::stop_reason::diverged_iters;

		return info;
	}


protected:
	template<class Op, class W, class T>
	void correct(int nr, Op & P, W & z, W & v, T & x) {
		if (params.pre_side == precond_side::right) {
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


	template<class T>
	void orthogonalize(T & v, int k) {
		auto & hessenberg = *hmat;
		// modified Gram-Schmidt
		for (int j = 0; j < k; j++) {
			const double h_jk = v.dot(basis[j]).get();
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

	void back_solve(int nr) {
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
	std::unique_ptr<real[]> hessenberg_data;
	using hessenberg_mat = util::mdcolex<real, 2>;
	std::unique_ptr<hessenberg_mat> hmat;
	util::span<typename iface::workvec_t> basis;
	std::vector<real> sinvec, cosvec;
	std::vector<real> dwvec, dyvec;
	settings params;
};
template<class V> solver(const settings&,V&&) -> solver<V>;

}

namespace flecsi::linalg {

template <class W, class... Ops>
struct traits<krylov_params<gmres::settings, W, Ops...>> {
	using op = krylov_interface<W, gmres::solver>;
};

}
#endif
