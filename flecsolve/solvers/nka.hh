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
#ifndef FLECSI_LINALG_OP_NKA_H
#define FLECSI_LINALG_OP_NKA_H

#include <flecsi/flog.hh>
#include <flecsi/util/array_ref.hh>

#include "krylov_parameters.hh"
#include "solver_settings.hh"

namespace flecsolve::nka {

enum workvecs : std::size_t { sol, res, correction, nwork };

}

namespace flecsolve::op {

template<class Params>
struct nka : base<Params,
                  typename Params::input_var_t,
                  typename Params::output_var_t> {
	using base_t = base<Params,
	                    typename Params::input_var_t,
	                    typename Params::output_var_t>;
	using real = typename Params::real;
	using base_t::params;

	nka(Params p) : base_t(std::move(p)),
	                substore(params.settings.max_dim + 1, ::flecsolve::nka::nwork) {
		reset();
	}

	const auto & get_operator() const { return params.A(); }

	void reset() {
		using namespace ::flecsolve::nka;
		params.settings.validate();
		const auto & settings = params.settings;
		auto & work = params.work;
		flog_assert(work.size() >= (settings.max_dim + 1) * 2 + nwork,
		            "NKA: not enough work vectors for specified max_dim");
		current_correction = 0;
		have_pending = false;
		have_subspace = false;
		subindex.reset(settings.max_dim + 1);
		if (static_cast<int>(substore.w.size()) < settings.max_dim + 1)
			substore = subspace_store(settings.max_dim + 1, nwork);
	}

	template<class DomainVec, class RangeVec>
	solve_info apply(const RangeVec & f, DomainVec & u) const {
		using namespace ::flecsolve::nka;

		solve_info info;

		const auto & settings = params.settings;

		auto & work = params.work;
		auto & sol = std::get<workvecs::sol>(work);
		auto & res = std::get<workvecs::res>(work);
		auto & correction = std::get<workvecs::correction>(work);

		auto & user_diagnostic = params.ops.diagnostic;

		const auto & F = params.A();
		const auto & P = params.P();

		F.residual(f, u, res);
		res.scale(-1.);

		auto res_norm = res.l2norm().get();
		info.res_norm_initial = res_norm;

		if (res_norm < settings.atol) {
			info.status = solve_info::stop_reason::converged_atol;
			return info;
		}

		sol.copy(u);

		auto pc_params = F.template get_parameters<op::label::jacobian>(sol);

		auto & pc_op = P.get_operator();
		// if using a frozen preconditioner set it up
		if (settings.freeze_pc) {
			pc_op.reset(pc_params);
		}
		for (int iter = 0; iter < settings.maxiter; iter++) {
			if (!settings.freeze_pc) {
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

			if (res_norm < settings.atol) {
				info.iters = iter + 1;
				info.status = solve_info::stop_reason::converged_atol;
				break;
			}
			if (res_norm < settings.rtol * info.res_norm_initial) {
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

	void relax() const {
		if (have_pending) {
			// drop the initial slot where pending vectors are stored.
			subindex.drop_first();
			have_pending = false;
		}
	}

	void factorize_normal_mat() const {
		// Solve the least squares problem using a Cholesky
		// factorization, dropping any vectors that
		// render the system nearly rank deficient
		// we'll first follow Carlson's implementation
		const auto & settings = params.settings;
		auto & h = substore.h;
		auto first = subindex.begin();

		// start the factorization at the entry indexed by first

		// Trivial initial factorization stage
		int nvec = 1;
		h[*first][*first] = 1.;

		for (auto k = first + 1; k != subindex.end(); ++k) {
			++nvec;

			if (nvec > settings.max_dim) {
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

			if (hkk > settings.angle_tol * settings.angle_tol) {
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
	std::vector<real> forward_backward_solve(const RangeVec & f) const {
		auto & work = params.work;

		std::vector<real> cv(params.settings.max_dim + 1, 0.);
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
	void compute_correction(RangeVec & f) const {
		auto & work = params.work;
		const auto & settings = params.settings;
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

			if (!settings.use_qr) {
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
			std::vector<real> cv(settings.max_dim + 1, 0.);

			if (!settings.use_qr) {
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

			if (settings.use_damping) {
				real eta = settings.damping_factor;
				if (settings.adaptive_damping) {
					eta = 1.0 - std::pow(0.9,
					                     std::min(current_correction,
					                              settings.max_dim));
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
			using vec_t = typename Params::workvec_t;
			vec_arr(size_t o, size_t l) : offset(o), len(l) {}
			flecsi::util::span<vec_t> span(typename Params::work_t & work) const {
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
	mutable subspace_index subindex;
	mutable subspace_store substore;
	mutable bool have_subspace, have_pending;
	mutable int current_correction;
};
template<class P>
nka(P) -> nka<P>;

} // namespace flecsolve::op


namespace flecsolve::nka {
struct settings : solver_settings {

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

struct options : solver_options {
	using settings_type = settings;
	using base_t = solver_options;

	options(const char * pre) : base_t(pre) {}

	auto operator()(settings_type & s) {
		auto desc = base_t::operator()(s);
		// clang-format off
		desc.add_options()
			(label("max-dim").c_str(), po::value<int>(&s.max_dim)->required(), "maximum dimension")
			(label("angle-tol").c_str(), po::value<double>(&s.angle_tol)->default_value(0.9), "angle tolerance")
			(label("freeze-pc").c_str(), po::value<bool>(&s.freeze_pc)->default_value(true), "freeze preconditioner")
			(label("use-qr").c_str(), po::value<bool>(&s.use_qr)->default_value(false), "use QR for factorization")
			(label("use-damping").c_str(), po::value<bool>(&s.use_damping)->default_value(false), "use damping")
			(label("adaptive-damping").c_str(), po::value<bool>(&s.adaptive_damping)->default_value(false), "use adaptive damping")
			(label("damping-factor").c_str(), po::value<double>(&s.damping_factor)->default_value(1.), "daping factor");
		// clang-format on
		return desc;
	}
};

template<std::size_t dim_bound = 10, std::size_t version = 0>
using topo_work = topo_work_base<nwork + 2 * (dim_bound + 1), version>;

template<std::size_t version>
struct dim_bound_t {
	static constexpr std::size_t value = version;
};

template<std::size_t V>
inline dim_bound_t<V> dim_bound;

struct work_factory {
	template<class Vec>
	constexpr auto operator()(Vec & b) const {
		return topo_work<>::get(b);
	}

	template<class Vec, std::size_t dbound, std::size_t Ver>
	constexpr auto operator()(dim_bound_t<dbound>, version_t<Ver>, Vec & b) const {
		return topo_work<dbound, Ver>::get(b);
	}

	template<class Vec, std::size_t dbound>
	constexpr auto operator()(dim_bound_t<dbound>, Vec & b) const {
		return topo_work<dbound, 0>::get(b);
	}
};

static inline work_factory make_work;

template<class Work>
struct solver : krylov_solver<op::nka, settings, Work>
{
	using base_t = krylov_solver<op::nka, settings, Work>;

	template<class W>
	solver(const settings & s, W && w) :
		base_t{s, std::forward<W>(w)} {}
};
template<class W>
solver(const settings &, W &&) -> solver<W>;

} // namespace flecsolve::nka

#endif
