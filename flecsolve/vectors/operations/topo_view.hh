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
#ifndef FLECSI_LINALG_VECTORS_OPERATIONS_TOPO_VIEW_HH
#define FLECSI_LINALG_VECTORS_OPERATIONS_TOPO_VIEW_HH

#include "flecsolve/util/future.hh"
#include "flecsolve/util/traits.hh"
#include "flecsolve/vectors/data/topo_view.hh"
#include "topo_tasks.hh"

namespace flecsolve::vec::ops {

template<class Data>
struct topo_view {
	using topo_t = typename Data::topo_t;
	using scalar = typename Data::scalar;
	static constexpr auto space = Data::space;
	using real = typename num_traits<scalar>::real;
	using len_t = flecsi::util::id;

	using vec_data = Data;

	using tasks = topo_tasks<vec_data, scalar, len_t>;

	static flecsi::scheduler & scheduler() {
		return *flecsi::scheduler::instance;
	}

	template<class Other>
	static void copy(const Other & x, vec_data & z) {
		static_assert(
			std::is_same_v<typename Other::topo_t, typename vec_data::topo_t>);
		static_assert(Other::space == vec_data::space);
		scheduler()
			.execute<tasks::template copy<
				typename Other::template acc_all<flecsi::ro>>>(
				flecsi::exec::on, x.topo(), z.ref(), x.ref());
	}

	static void zero(vec_data & x) {
		scheduler().execute<tasks::set_to_scalar>(
			flecsi::exec::on, x.topo(), x.ref(), 0.0);
	}

	static void set_random(vec_data & x, unsigned seed) {
		scheduler().execute<tasks::set_random>(x.topo(), x.ref(), seed);
	}

	static void set_to_scalar(scalar alpha, vec_data & x) {
		scheduler().execute<tasks::set_to_scalar>(
			flecsi::exec::on, x.topo(), x.ref(), alpha);
	}

	template<class F, class T>
	static void set_to_scalar(future_transform<future<T>, F> alpha,
	                          vec_data & x) {
		static_assert(
			std::is_convertible_v<T, scalar>,
			"set_to_scalar: future type must be convertible to scalar");
		scheduler().execute<tasks::template set_to_scalar_future<F, T>>(
			flecsi::exec::on, x.topo(), x.ref(), alpha.fut, alpha.f);
	}

	static void scale(scalar alpha, vec_data & x) {
		scheduler().execute<tasks::scale_self>(
			flecsi::exec::on, x.topo(), x.ref(), alpha);
	}

	template<class F, class T>
	static void scale(future_transform<future<T>, F> alpha, vec_data & x) {
		static_assert(std::is_convertible_v<T, scalar>,
		              "scale: future type must be convertible to scalar");
		scheduler().execute<tasks::template scale_self_future<F, T>>(
			flecsi::exec::on, x.topo(), x.ref(), alpha.fut, alpha.f);
	}

	static void scale(scalar alpha, const vec_data & x, vec_data & y) {
		flog_assert(x.fid() != y.fid(),
		            "scale operation: vector data cannot be the same");
		scheduler().execute<tasks::scale>(
			flecsi::exec::on, x.topo(), x.ref(), y.ref(), alpha);
	}

	template<class F, class T>
	static void scale(future_transform<future<T>, F> alpha,
	                  const vec_data & x,
	                  vec_data & y) {
		static_assert(std::is_convertible_v<T, scalar>,
		              "scale: future type must be convertible to scalar");
		flog_assert(x.fid() != y.fid(),
		            "scale operation: vector data cannot be the same");
		scheduler().execute<tasks::template scale_future<F, T>>(
			flecsi::exec::on, x.topo(), x.ref(), y.ref(), alpha.fut, alpha.f);
	}

	static void add(const vec_data & x, const vec_data & y, vec_data & z) {
		if (x.fid() == z.fid()) {
			scheduler().execute<tasks::add_self>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref());
		}
		else if (y.fid() == z.fid()) {
			scheduler().execute<tasks::add_self>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref());
		}
		else {
			scheduler().execute<tasks::add>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	static void subtract(const vec_data & x, const vec_data & y, vec_data & z) {
		if (x.fid() == z.fid()) {
			scheduler().execute<tasks::template subtract_self<true>>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref());
		}
		else if (y.fid() == z.fid()) {
			scheduler().execute<tasks::template subtract_self<false>>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref());
		}
		else {
			scheduler().execute<tasks::subtract>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	static void multiply(const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			scheduler().execute<tasks::multiply_self>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref());
		}
		else if (z.fid() == y.fid()) {
			scheduler().execute<tasks::multiply_self>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref());
		}
		else {
			scheduler().execute<tasks::multiply>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	static void divide(const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			scheduler().execute<tasks::template divide_self<true>>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref());
		}
		else if (z.fid() == y.fid()) {
			scheduler().execute<tasks::template divide_self<false>>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref());
		}
		else {
			scheduler().execute<tasks::divide>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	static void reciprocal(const vec_data & x, vec_data & y) {
		if (x.fid() == y.fid()) {
			scheduler().execute<tasks::reciprocal_self>(
				flecsi::exec::on, y.topo(), y.ref());
		}
		else {
			scheduler().execute<tasks::reciprocal>(
				flecsi::exec::on, y.topo(), y.ref(), x.ref());
		}
	}

	static void linear_sum(scalar alpha,
	                       const vec_data & x,
	                       scalar beta,
	                       const vec_data & y,
	                       vec_data & z) {
		if (z.fid() == x.fid()) {
			scheduler().execute<tasks::template linear_sum_self<true>>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref(), alpha, beta);
		}
		else if (z.fid() == y.fid()) {
			scheduler().execute<tasks::template linear_sum_self<false>>(
				flecsi::exec::on, z.topo(), x.ref(), z.ref(), alpha, beta);
		}
		else {
			scheduler().execute<tasks::linear_sum>(flecsi::exec::on,
			                                       z.topo(),
			                                       z.ref(),
			                                       alpha,
			                                       x.ref(),
			                                       beta,
			                                       y.ref());
		}
	}

	template<class F, class T>
	static void linear_sum(future_transform<future<T>, F> alpha,
	                       const vec_data & x,
	                       scalar beta,
	                       const vec_data & y,
	                       vec_data & z) {
		static_assert(std::is_convertible_v<T, scalar>,
		              "linear_sum: future type must be convertible to scalar");
		if (z.fid() == x.fid()) {
			scheduler()
				.execute<
					tasks::template linear_sum_self_alpha_future<true, F, T>>(
					flecsi::exec::on,
					z.topo(),
					z.ref(),
					y.ref(),
					alpha.fut,
					alpha.f,
					beta);
		}
		else if (z.fid() == y.fid()) {
			scheduler()
				.execute<
					tasks::template linear_sum_self_alpha_future<false, F, T>>(
					flecsi::exec::on,
					z.topo(),
					x.ref(),
					z.ref(),
					alpha.fut,
					alpha.f,
					beta);
		}
		else {
			scheduler().execute<tasks::template linear_sum_alpha_future<F, T>>(
				flecsi::exec::on,
				z.topo(),
				z.ref(),
				alpha.fut,
				alpha.f,
				x.ref(),
				beta,
				y.ref());
		}
	}

	template<class F, class T>
	static void linear_sum(scalar alpha,
	                       const vec_data & x,
	                       future_transform<future<T>, F> beta,
	                       const vec_data & y,
	                       vec_data & z) {
		static_assert(std::is_convertible_v<T, scalar>,
		              "linear_sum: future type must be convertible to scalar");
		if (z.fid() == x.fid()) {
			scheduler()
				.execute<
					tasks::template linear_sum_self_beta_future<true, F, T>>(
					flecsi::exec::on,
					z.topo(),
					z.ref(),
					y.ref(),
					alpha,
					beta.fut,
					beta.f);
		}
		else if (z.fid() == y.fid()) {
			scheduler()
				.execute<
					tasks::template linear_sum_self_beta_future<false, F, T>>(
					flecsi::exec::on,
					z.topo(),
					x.ref(),
					z.ref(),
					alpha,
					beta.fut,
					beta.f);
		}
		else {
			scheduler().execute<tasks::template linear_sum_beta_future<F, T>>(
				flecsi::exec::on,
				z.topo(),
				z.ref(),
				alpha,
				x.ref(),
				beta.fut,
				beta.f,
				y.ref());
		}
	}

	template<class AlphaF, class AlphaT, class BetaF, class BetaT>
	static void linear_sum(future_transform<future<AlphaT>, AlphaF> alpha,
	                       const vec_data & x,
	                       future_transform<future<BetaT>, BetaF> beta,
	                       const vec_data & y,
	                       vec_data & z) {
		static_assert(std::is_convertible_v<AlphaT, scalar>,
		              "linear_sum: future type must be convertible to scalar");
		static_assert(std::is_convertible_v<BetaT, scalar>,
		              "linear_sum: future type must be convertible to scalar");
		if (z.fid() == x.fid()) {
			scheduler()
				.execute<tasks::template linear_sum_self_future<true,
			                                                    AlphaF,
			                                                    AlphaT,
			                                                    BetaF,
			                                                    BetaT>>(
					flecsi::exec::on,
					z.topo(),
					z.ref(),
					y.ref(),
					alpha.fut,
					alpha.f,
					beta.fut,
					beta.f);
		}
		else if (z.fid() == y.fid()) {
			scheduler()
				.execute<tasks::template linear_sum_self_future<false,
			                                                    AlphaF,
			                                                    AlphaT,
			                                                    BetaF,
			                                                    BetaT>>(
					flecsi::exec::on,
					z.topo(),
					x.ref(),
					z.ref(),
					alpha.fut,
					alpha.f,
					beta.fut,
					beta.f);
		}
		else {
			scheduler()
				.execute<tasks::template linear_sum_future<AlphaF,
			                                               AlphaT,
			                                               BetaF,
			                                               BetaT>>(
					flecsi::exec::on,
					z.topo(),
					z.ref(),
					alpha.fut,
					alpha.f,
					x.ref(),
					beta.fut,
					beta.f,
					y.ref());
		}
	}

	static void
	axpy(scalar alpha, const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			scheduler().execute<tasks::template axpy_self<true>>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref(), alpha);
		}
		else if (z.fid() == y.fid()) {
			scheduler().execute<tasks::template axpy_self<false>>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref(), alpha);
		}
		else {
			scheduler().execute<tasks::axpy>(
				flecsi::exec::on, z.topo(), z.ref(), alpha, x.ref(), y.ref());
		}
	}

	template<class F, class T>
	static void axpy(future_transform<future<T>, F> alpha,
	                 const vec_data & x,
	                 const vec_data & y,
	                 vec_data & z) {
		static_assert(std::is_convertible_v<T, scalar>,
		              "axpy: future type must be convertible to scalar");
		if (z.fid() == x.fid()) {
			scheduler().execute<tasks::template axpy_self_future<true, F, T>>(
				flecsi::exec::on,
				z.topo(),
				z.ref(),
				y.ref(),
				alpha.fut,
				alpha.f);
		}
		else if (z.fid() == y.fid()) {
			scheduler().execute<tasks::template axpy_self_future<false, F, T>>(
				flecsi::exec::on,
				z.topo(),
				z.ref(),
				x.ref(),
				alpha.fut,
				alpha.f);
		}
		else {
			scheduler().execute<tasks::template axpy_future<F, T>>(
				flecsi::exec::on,
				z.topo(),
				z.ref(),
				alpha.fut,
				alpha.f,
				x.ref(),
				y.ref());
		}
	}

	static void
	axpby(scalar alpha, scalar beta, const vec_data & x, vec_data & z) {
		scheduler().execute<tasks::axpby>(
			flecsi::exec::on, z.topo(), z.ref(), x.ref(), alpha, beta);
	}

	template<class F, class T>
	static void axpby(future_transform<future<T>, F> alpha,
	                  scalar beta,
	                  const vec_data & x,
	                  vec_data & z) {
		static_assert(std::is_convertible_v<T, scalar>,
		              "axpby: future type must be convertible to scalar");
		scheduler().execute<tasks::template axpby_alpha_future<F, T>>(
			flecsi::exec::on,
			z.topo(),
			z.ref(),
			x.ref(),
			alpha.fut,
			alpha.f,
			beta);
	}

	template<class F, class T>
	static void axpby(scalar alpha,
	                  future_transform<future<T>, F> beta,
	                  const vec_data & x,
	                  vec_data & z) {
		static_assert(std::is_convertible_v<T, scalar>,
		              "axpby: future type must be convertible to scalar");
		scheduler().execute<tasks::template axpby_beta_future<F, T>>(
			flecsi::exec::on,
			z.topo(),
			z.ref(),
			x.ref(),
			alpha,
			beta.fut,
			beta.f);
	}

	template<class AlphaF, class AlphaT, class BetaF, class BetaT>
	static void axpby(future_transform<future<AlphaT>, AlphaF> alpha,
	                  future_transform<future<BetaT>, BetaF> beta,
	                  const vec_data & x,
	                  vec_data & z) {
		static_assert(std::is_convertible_v<AlphaT, scalar>,
		              "axpby: future type must be convertible to scalar");
		static_assert(std::is_convertible_v<BetaT, scalar>,
		              "axpby: future type must be convertible to scalar");
		scheduler()
			.execute<
				tasks::template axpby_future<AlphaF, AlphaT, BetaF, BetaT>>(
				flecsi::exec::on,
				z.topo(),
				z.ref(),
				x.ref(),
				alpha.fut,
				alpha.f,
				beta.fut,
				beta.f);
	}

	static void abs(const vec_data & x, vec_data & y) {
		if (y.fid() == x.fid()) {
			scheduler().execute<tasks::abs_self>(
				flecsi::exec::on, y.topo(), y.ref());
		}
		else {
			scheduler().execute<tasks::abs>(
				flecsi::exec::on, y.topo(), y.ref(), x.ref());
		}
	}

	static void add_scalar(const vec_data & x, scalar alpha, vec_data & y) {
		if (x.fid() == y.fid()) {
			scheduler().execute<tasks::add_scalar_self>(
				flecsi::exec::on, y.topo(), y.ref(), alpha);
		}
		else {
			scheduler().execute<tasks::add_scalar>(
				flecsi::exec::on, y.topo(), y.ref(), x.ref(), alpha);
		}
	}

	template<class F, class T>
	static void add_scalar(const vec_data & x,
	                       future_transform<future<T>, F> alpha,
	                       vec_data & y) {
		static_assert(std::is_convertible_v<T, scalar>,
		              "add_scalar: future type must be convertible to scalar");
		if (x.fid() == y.fid()) {
			scheduler().execute<tasks::template add_scalar_self_future<F, T>>(
				flecsi::exec::on, y.topo(), y.ref(), alpha.fut, alpha.f);
		}
		else {
			scheduler().execute<tasks::template add_scalar_future<F, T>>(
				flecsi::exec::on,
				y.topo(),
				y.ref(),
				x.ref(),
				alpha.fut,
				alpha.f);
		}
	}

	static auto min(const vec_data & x) {
		return scheduler().reduce<tasks::local_min, flecsi::exec::fold::min>(
			flecsi::exec::on, x.topo(), x.ref());
	}

	static auto max(const vec_data & y) {
		return scheduler().reduce<tasks::local_max, flecsi::exec::fold::max>(
			flecsi::exec::on, y.topo(), y.ref());
	}

	template<unsigned short p>
	static auto lp_norm_local(const vec_data & x) {
		if constexpr (p == 1) {
			return scheduler()
			    .reduce<tasks::l1_norm_local, flecsi::exec::fold::sum>(
					flecsi::exec::on, x.topo(), x.ref());
		}
		else if constexpr (p == 2) {
			return scheduler()
			    .reduce<tasks::l2_norm_local, flecsi::exec::fold::sum>(
					flecsi::exec::on, x.topo(), x.ref());
		}
		else {
			return scheduler()
			    .reduce<tasks::lp_norm_local, flecsi::exec::fold::sum>(
					flecsi::exec::on, x.topo(), x.ref(), p)
			    .get();
		}
	}

	template<unsigned short p>
	static auto lp_norm(const vec_data & x) {
		auto fut = lp_norm_local<p>(x);
		if constexpr (p == 1) {
			return fut;
		}
		else if constexpr (p == 2) {
			return future_transform{std::move(fut),
			                        [](auto v) { return std::sqrt(v); }};
		}
		else {
			return future_transform{std::move(fut),
			                        [](auto v) { return std::pow(v, 1. / p); }};
		}
	}

	static auto inf_norm(const vec_data & x) {
		return scheduler()
		    .reduce<tasks::inf_norm_local, flecsi::exec::fold::max>(
				flecsi::exec::on, x.topo(), x.ref());
	}

	static auto dot(const vec_data & x, const vec_data & y) {
		return scheduler().reduce<tasks::scalar_prod, flecsi::exec::fold::sum>(
			flecsi::exec::on, x.topo(), x.ref(), y.ref());
	}

	static auto global_size(const vec_data & x) {
		return scheduler().reduce<tasks::local_size, flecsi::exec::fold::sum>(
			x.topo());
	}

	static len_t local_size(const vec_data & x) {
		len_t length;
		scheduler().execute<tasks::get_local_size, flecsi::mpi>(x.topo(),
		                                                        &length);
		return length;
	}

	static void dump(std::string_view pre, const vec_data & x) {
		// TODO: update for multiaccessor
		scheduler().execute<tasks::dump>(
			flecsi::exec::on, pre, x.topo(), x.ref());
	}

	template<class F, class... Vecs>
	static constexpr decltype(auto) apply(F && f, Vecs &&... vecs) {
		return std::forward<F>(f)(std::forward<Vecs>(vecs)...);
	}
};

}

#endif
