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

	template<class Other>
	static void copy(const Other & x, vec_data & z) {
		static_assert(
			std::is_same_v<typename Other::topo_t, typename vec_data::topo_t>);
		static_assert(Other::space == vec_data::space);
		flecsi::execute<
			tasks::template copy<typename Other::template acc_all<flecsi::ro>>>(
				flecsi::exec::on, x.topo(), z.ref(), x.ref());
	}

	static void zero(vec_data & x) {
		flecsi::execute<tasks::set_to_scalar>(
			flecsi::exec::on, x.topo(), x.ref(), 0.0);
	}

	static void set_random(vec_data & x, unsigned seed) {
		flecsi::execute<tasks::set_random>(x.topo(), x.ref(), seed);
	}

	static void set_to_scalar(scalar alpha, vec_data & x) {
		flecsi::execute<tasks::set_to_scalar>(
			flecsi::exec::on, x.topo(), x.ref(), alpha);
	}

	static void scale(scalar alpha, vec_data & x) {
		flecsi::execute<tasks::scale_self>(
			flecsi::exec::on, x.topo(), x.ref(), alpha);
	}

	static void scale(scalar alpha, const vec_data & x, vec_data & y) {
		flog_assert(x.fid() != y.fid(),
		            "scale operation: vector data cannot be the same");
		flecsi::execute<tasks::scale>(
			flecsi::exec::on, x.topo(), x.ref(), y.ref(), alpha);
	}

	static void add(const vec_data & x, const vec_data & y, vec_data & z) {
		if (x.fid() == z.fid()) {
			flecsi::execute<tasks::add_self>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref());
		}
		else if (y.fid() == z.fid()) {
			flecsi::execute<tasks::add_self>(
				flecsi::exec::on,z.topo(), z.ref(), x.ref());
		}
		else {
			flecsi::execute<tasks::add>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	static void subtract(const vec_data & x, const vec_data & y, vec_data & z) {
		if (x.fid() == z.fid()) {
			flecsi::execute<tasks::template subtract_self<true>>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref());
		}
		else if (y.fid() == z.fid()) {
			flecsi::execute<tasks::template subtract_self<false>>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref());
		}
		else {
			flecsi::execute<tasks::subtract>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	static void multiply(const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			flecsi::execute<tasks::multiply_self>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref());
		}
		else if (z.fid() == y.fid()) {
			flecsi::execute<tasks::multiply_self>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref());
		}
		else {
			flecsi::execute<tasks::multiply>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	static void divide(const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			flecsi::execute<tasks::template divide_self<true>>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref());
		}
		else if (z.fid() == y.fid()) {
			flecsi::execute<tasks::template divide_self<false>>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref());
		}
		else {
			flecsi::execute<tasks::divide>(
				flecsi::exec::on, z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	static void reciprocal(const vec_data & x, vec_data & y) {
		if (x.fid() == y.fid()) {
			flecsi::execute<tasks::reciprocal_self>(
				flecsi::exec::on, y.topo(), y.ref());
		}
		else {
			flecsi::execute<tasks::reciprocal>(
				flecsi::exec::on, y.topo(), y.ref(), x.ref());
		}
	}

	static void linear_sum(scalar alpha,
	                       const vec_data & x,
	                       scalar beta,
	                       const vec_data & y,
	                       vec_data & z) {
		if (z.fid() == x.fid()) {
			flecsi::execute<tasks::template linear_sum_self<true>>(
				flecsi::exec::on, z.topo(), z.ref(), y.ref(), alpha, beta);
		}
		else if (z.fid() == y.fid()) {
			flecsi::execute<tasks::template linear_sum_self<false>>(
				flecsi::exec::on, z.topo(), x.ref(), z.ref(), alpha, beta);
		}
		else {
			flecsi::execute<tasks::linear_sum>(
				flecsi::exec::on, z.topo(), z.ref(), alpha, x.ref(), beta, y.ref());
		}
	}

	static void
	axpy(scalar alpha, const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			flecsi::execute<tasks::template axpy_self<true>>(
				                flecsi::exec::on, z.topo(), z.ref(), y.ref(), alpha);
		}
		else if (z.fid() == y.fid()) {
			flecsi::execute<tasks::template axpy_self<false>>(
				                flecsi::exec::on, z.topo(), z.ref(), x.ref(), alpha);
		}
		else {
			flecsi::execute<tasks::axpy>(
				flecsi::exec::on, z.topo(), z.ref(), alpha, x.ref(), y.ref());
		}
	}

	static void
	axpby(scalar alpha, scalar beta, const vec_data & x, vec_data & z) {
		flecsi::execute<tasks::axpby>(
			flecsi::exec::on, z.topo(), z.ref(), x.ref(), alpha, beta);
	}

	static void abs(const vec_data & x, vec_data & y) {
		if (y.fid() == x.fid()) {
			flecsi::execute<tasks::abs_self>(
				flecsi::exec::on, y.topo(), y.ref());
		}
		else {
			flecsi::execute<tasks::abs>(
				flecsi::exec::on, y.topo(), y.ref(), x.ref());
		}
	}

	static void add_scalar(const vec_data & x, scalar alpha, vec_data & y) {
		if (x.fid() == y.fid()) {
			flecsi::execute<tasks::add_scalar_self>(
				flecsi::exec::on, y.topo(), y.ref(), alpha);
		}
		else {
			flecsi::execute<tasks::add_scalar>(
				flecsi::exec::on, y.topo(), y.ref(), x.ref(), alpha);
		}
	}

	static auto min(const vec_data & x) {
		return flecsi::reduce<tasks::local_min,
		                      flecsi::exec::fold::min>(
			                      flecsi::exec::on, x.topo(), x.ref());
	}

	static auto max(const vec_data & y) {
		return flecsi::reduce<tasks::local_max,
		                      flecsi::exec::fold::max>(
			                      flecsi::exec::on, y.topo(), y.ref());
	}

	template<unsigned short p>
	static auto lp_norm_local(const vec_data & x) {
		if constexpr (p == 1) {
			return flecsi::reduce<tasks::l1_norm_local,
			                      flecsi::exec::fold::sum>(
				                      flecsi::exec::on, x.topo(), x.ref());
		}
		else if constexpr (p == 2) {
			return flecsi::reduce<tasks::l2_norm_local,
			                      flecsi::exec::fold::sum>(flecsi::exec::on,
			                                               x.topo(), x.ref());
		}
		else {
			return flecsi::reduce<tasks::lp_norm_local,
			                      flecsi::exec::fold::sum>(
				                      flecsi::exec::on, x.topo(), x.ref(), p).get();
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
		return flecsi::reduce<tasks::inf_norm_local,
		                      flecsi::exec::fold::max>(
			                      flecsi::exec::on, x.topo(), x.ref());
	}

	static auto dot(const vec_data & x, const vec_data & y) {
		return flecsi::reduce<tasks::scalar_prod,
		                      flecsi::exec::fold::sum>(
			                      flecsi::exec::on, x.topo(), x.ref(), y.ref());
	}

	static auto global_size(const vec_data & x) {
		return flecsi::reduce<tasks::local_size,
		                      flecsi::exec::fold::sum>(x.topo());
	}

	static len_t local_size(const vec_data & x) {
		len_t length;
		flecsi::execute<tasks::get_local_size, flecsi::mpi>(x.topo(), &length);
		return length;
	}

	static void dump(std::string_view pre, const vec_data & x) {
		// TODO: update for multiaccessor
		flecsi::execute<tasks::dump>(flecsi::exec::on, pre, x.topo(), x.ref());
	}

	template<class F, class... Vecs>
	static constexpr decltype(auto) apply(F && f, Vecs &&... vecs) {
		return std::forward<F>(f)(std::forward<Vecs>(vecs)...);
	}
};

}

#endif
