#ifndef FLECSI_LINALG_OP_NKA_H
#define FLECSI_LINALG_OP_NKA_H

#include <flecsi/flog.hh>
#include <flecsi/util/array_ref.hh>

#include "krylov_interface.hh"
#include "solver_settings.hh"

namespace flecsolve::nka {

enum workvecs : std::size_t { sol, res, correction, nwork };

struct settings : solver_settings {
	using base = solver_settings;

	settings(int maxiter, float rtol, float atol, int max_dim, float angle_tol)
		: base{maxiter, rtol, atol, false}, max_dim{max_dim},
		  angle_tol{angle_tol}, freeze_pc{true}, use_qr{false},
		  use_damping{false}, adaptive_damping{false}, damping_factor(1.0) {}

	void validate() {
		if (adaptive_damping) {
			flog_assert(use_damping,
			            "NKA: damping must be enabled for adaptive_damping");
		}
		flog_assert(max_dim > 0, "NKA: maximum dimension must be > 0.");
		flog_assert(angle_tol > 0., "NKA: angle tolerance must be > 0.0.");
	}

	int max_dim;
	double angle_tol;
	int max_fun_evals;
	bool freeze_pc;
	bool use_qr;
	bool use_damping;
	bool adaptive_damping;
	double damping_factor;
};

template<std::size_t dim_bound = 10, std::size_t version = 0>
using topo_work = topo_work_base<nwork + 2 * (dim_bound + 1), version>;

template<class Workspace>
struct solver : krylov_interface<Workspace, solver> {
	using settings_type = settings;
	using iface = krylov_interface<Workspace, solver>;
	using real = typename iface::real;
	using iface::work;

	template<class V>
	solver(const settings & params, V && workspace)
		: iface{std::forward<V>(workspace)}, params(params),
		  substore(params.max_dim + 1, nwork) {
		reset();
	}

	void reset(const settings & params) {
		this->params = params;
		reset();
	}

	void reset() {
		this->params.validate();
		flog_assert(work.size() >= (params.max_dim + 1) * 2 + nwork,
		            "NKA: not enough work vectors for specified max_dim");
		current_correction = 0;
		have_pending = false;
		have_subspace = false;
		subindex.reset(params.max_dim + 1);
		if (static_cast<int>(substore.w.size()) < params.max_dim + 1)
			substore = subspace_store(params.max_dim + 1, nwork);
	}

	template<class Op,
	         class DomainVec,
	         class RangeVec,
	         class Precond,
	         class Diag>
	solve_info apply(const Op & F,
	                 const RangeVec & f,
	                 DomainVec & u,
	                 Precond & P,
	                 Diag && user_diagnostic) {
		solve_info info;

		auto & sol = std::get<workvecs::sol>(work);
		auto & res = std::get<workvecs::res>(work);
		auto & correction = std::get<workvecs::correction>(work);

		F.residual(f, u, res);
		res.scale(-1.);

		auto res_norm = res.l2norm().get();
		info.res_norm_initial = res_norm;

		if (res_norm < params.atol) {
			info.status = solve_info::stop_reason::converged_atol;
			return info;
		}

		sol.copy(u);

		auto pc_params = F.template get_parameters<op::label::jacobian>(sol);

		auto & pc_op = P.get_operator();
		// if using a frozen preconditioner set it up
		if (params.freeze_pc) {
			pc_op.reset(pc_params);
		}
		for (int iter = 0; iter < params.maxiter; iter++) {
			if (!params.freeze_pc) {
				pc_params = F.template get_parameters<op::label::jacobian>(sol);
				pc_op.reset(pc_params);
			}
			P.apply(res, correction);

			compute_correction(correction);

			// apply correction
			sol.axpy(-1., correction, sol);

			F.residual(f, sol, res);
			res.scale(-1.);

			res_norm = res.l2norm().get();

			if (res_norm < params.atol) {
				info.iters = iter + 1;
				info.status = solve_info::stop_reason::converged_atol;
				break;
			}
			if (res_norm < params.rtol * info.res_norm_initial) {
				info.iters = iter + 1;
				info.status = solve_info::stop_reason::converged_rtol;
				break;
			}
			if (user_diagnostic(sol, res_norm)) {
				info.status = solve_info::stop_reason::converged_user;
				info.iters = iter + 1;
				break;
			}
		}

		info.res_norm_final = res_norm;
		info.sol_norm_final = sol.l2norm().get();

		u.copy(sol);
		return info;
	}

	void relax() {
		if (have_pending) {
			// drop the initial slot where pending vectors are stored.
			subindex.drop_first();
			have_pending = false;
		}
	}

	void factorize_normal_mat() {
		// Solve the least squares problem using a Cholesky
		// factorization, dropping any vectors that
		// render the system nearly rank deficient
		// we'll first follow Carlson's implementation

		auto & h = substore.h;
		auto first = subindex.begin();

		// start the factorization at the entry indexed by first

		// Trivial initial factorization stage
		int nvec = 1;
		h[*first][*first] = 1.;

		for (auto k = first + 1; k != subindex.end(); ++k) {
			++nvec;

			if (nvec > params.max_dim) {
				flog_assert(*k == subindex.last, "NKA: list error");
				subindex.pop();
				break;
			}

			// Single stage of Choleski factorization

			auto hk = h[*k];
			real hkk = 1.;
			for (auto j = subindex.begin(); j != k; ++j) {
				auto hj = h[*j];
				real hkj = hj[*k];
				for (auto i = subindex.begin(); i != j; ++i) {
					hkj -= hk[*i] * hj[*i];
				}
				hkj /= hj[*j];
				hk[*j] = hkj;
				hkk -= hkj * hkj;
			}

			if (hkk > params.angle_tol * params.angle_tol) {
				hk[*k] = std::sqrt(hkk);
			}
			else {
				// The current w nearly lies in the span of the pervious vectors
				// so drop it.
				subindex.drop(*k);
				// back up and move on to the next vector
				--k;
				--nvec;
			}
		}
	}

	template<class RangeVec>
	std::vector<real> forward_backward_solve(const RangeVec & f) {
		std::vector<real> cv(params.max_dim + 1, 0.);
		auto & h = substore.h;
		auto w = substore.w.span(work);

		// project f onto the span of the w vectors
		// forward substitution
		for (int j : subindex) {
			real cj = f.dot(w[j]).get();
			for (int i : subindex.range(j)) {
				cj -= h[j][i] * cv[i];
			}

			cv[j] = cj / h[j][j];
		}

		// backward substitution
		for (int j : subindex.reverse()) {
			real cj = cv[j];
			for (int i : subindex.reverse(j)) {
				cj -= h[i][j] * cv[i];
			}
			cv[j] = cj / h[j][j];
		}

		return cv;
	}

	template<class RangeVec>
	void compute_correction(RangeVec & f) {
		++current_correction;

		auto & h = substore.h;
		auto w_arr = substore.w.span(work);
		auto v_arr = substore.v.span(work);
		// update acceleration subspace
		if (have_pending) {
			// next function difference w_1
			auto & w = w_arr[subindex.first];

			w.axpy(-1., f, w);

			auto s = w.l2norm().get();
			/* If the function difference is 0, we can't update the subspace
			   with this data; so we toss it out and continue.  In this case it
			   is likely that the outer iterative solution procedure has gone
			   badly awry (unless the function value is itself 0), and we merely
			   want to do something reasonable here and hope that situation is
			   detected on the outside. */
			if (s == 0.0) {
				flog_warn(
					"NKA: current vector not valid!!, relax() being called.");
				relax();
			}

			auto & v = v_arr[subindex.first];

			// normalize w_1 and apply same factor to v_1
			w.scale(1. / s);
			v.scale(1. / s);

			if (!params.use_qr) {
				// update H.
				for (auto k = subindex.begin() + 1; k != subindex.end(); ++k) {
					h[subindex.first][*k] = w.dot(w_arr[*k]).get();
				}

				/*  Choleski factorization of H = W^t W
				    original matrix kept in the upper triangle (implicit unit
				    diagonal) lower triangle holds factorization */
				factorize_normal_mat();
			}
			else {
				flog_error("NKA: QR factorization not implemented");
			}
			// indicate we have a subspace
			have_subspace = true;
			have_pending = false;
		}

		// accelerated correction
		// locate storage location for the new vector by finding first free
		// location in list
		int new_loc = subindex.pop_free();

		/* store f in currently free location of w, so that we can
		   update it to the difference on the next iteration. */
		w_arr[new_loc].copy(f);

		if (have_subspace) {
			// create a row vector to store the solution components for the
			// correction vector
			std::vector<real> cv(params.max_dim + 1, 0.);

			if (!params.use_qr) {
				cv = forward_backward_solve(f);
			}
			else {
				flog_error("NKA: QR factorization not implemented.");
			}

			/* at this point the solution to the minimization problem has
			   been computed and stored in cv, we now compute the accelerated
			   correction. */
			for (int k : subindex) {
				f.axpy(cv[k], v_arr[k], f);
				f.axpy(-cv[k], w_arr[k], f);
			}

			if (params.use_damping) {
				real eta = params.damping_factor;
				if (params.adaptive_damping) {
					eta = 1.0 - std::pow(0.9,
					                     std::min(current_correction,
					                              params.max_dim));
				}

				// scale the residual vector
				f.scale(eta, f);
			}
		}

		// save the correction, accelerated or otherwise for the next call in
		// the v matrix
		v_arr[new_loc].copy(f);

		subindex.push(new_loc);

		// The original f and accelerated correction are cached for the next
		// call.
		have_pending = true;
	}

protected:
	struct subspace_index {
		static constexpr int EOL = -1;

		void reset(int n) {
			first = EOL;
			last = EOL;
			free = 0;
			next.resize(n);
			prev.resize(n);

			for (int k = 0; k < n - 1; ++k) {
				next[k] = k + 1;
			}
			next[n - 1] = EOL;
		}

		int pop_free() {
			flog_assert(free != EOL, "NKA: free list error");
			int ret = free;
			free = next[free];
			return ret;
		}

		// Drop first location in list
		void drop_first() {
			flog_assert(first >= 0, "NKA: list error");
			int new_loc = first;
			first = next[first];
			if (first == EOL)
				last = EOL;
			else
				prev[first] = EOL;

			next[new_loc] = free;
			free = new_loc;
		}

		void drop(int loc) {
			flog_assert(prev[loc] != EOL, "NKA: list error");
			next[prev[loc]] = next[loc];
			if (next[loc] == EOL)
				last = prev[loc];
			else
				prev[next[loc]] = prev[loc];

			next[loc] = free;
			free = loc;
		}

		void pop() {
			int k = last;
			next[last] = free;
			free = k;
			last = prev[k];
			next[last] = EOL;
		}

		void push(int loc) {
			// prepend
			prev[loc] = EOL;
			next[loc] = first;
			if (first == EOL)
				last = loc;
			else
				prev[first] = loc;

			first = loc;
		}

		struct iterator {
			int k;
			const std::vector<int> & next;
			const std::vector<int> & prev;
			constexpr iterator & operator++() {
				k = next[k];
				return *this;
			}

			constexpr iterator operator++(int) {
				const iterator ret = *this;
				++*this;
				return ret;
			}

			constexpr iterator & operator--() {
				k = prev[k];
				return *this;
			}

			constexpr iterator operator+(std::size_t n) const {
				iterator ret = *this;
				for (std::size_t i = 0; i < n; ++i)
					++ret;
				return ret;
			}

			constexpr const int & operator*() const { return k; }
			constexpr bool operator!=(const iterator & o) const noexcept {
				return k != o.k;
			}
		};

		iterator begin() const noexcept { return {first, next, prev}; }

		iterator end() const noexcept { return {EOL, next, prev}; }

		struct subr {
			int b, e;
			const std::vector<int> & prev;
			const std::vector<int> & next;
			iterator begin() const noexcept { return {b, next, prev}; }
			iterator end() const noexcept { return {e, next, prev}; }
		};

		subr range(int end) { return {first, end, prev, next}; }

		subr range(int beg, int end) { return {beg, end, prev, next}; }

		subr reverse(int end = EOL) { return {last, end, next, prev}; }

		int first, last, free;
		std::vector<int> prev;
		std::vector<int> next;
	};
	struct mat {
		using view = flecsi::util::mdspan<real, 2>;
		using size_type = typename view::size_type;

		mat(size_type m, size_type n)
			: data(std::make_unique<real[]>(m * n)), mspan(data.get(), {m, n}) {
		}

		constexpr decltype(auto) operator[](size_type i) noexcept {
			return mspan[i];
		}

	protected:
		std::unique_ptr<real[]> data;
		view mspan;
	};
	struct subspace_store {
		struct vec_arr {
			using size_type = std::size_t;
			using vec_t = typename iface::workvec_t;
			vec_arr(size_t o, size_t l) : offset(o), len(l) {}
			flecsi::util::span<vec_t> span(Workspace & work) const {
				return {work.data() + offset, len};
			}

			size_type size() { return len; }

		protected:
			size_type offset;
			size_type len;
		};

		subspace_store(int n, std::size_t off)
			: h(n, n), w(off, n), v(off + n, n) {}

		mat h;
		vec_arr w, v;
	};
	settings params;
	subspace_index subindex;
	subspace_store substore;
	bool have_subspace, have_pending;
	int current_correction;
};
template<class V>
solver(const settings &, V &&) -> solver<V>;
}

namespace flecsolve {

template<class W, class... Ops>
struct traits<krylov_params<nka::settings, W, Ops...>> {
	using op = krylov_interface<W, nka::solver>;
};

}

#endif
