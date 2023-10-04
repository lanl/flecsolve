#ifndef FLECSI_LINALG_VECTORS_OPERATIONS_TOPO_VIEW_HH
#define FLECSI_LINALG_VECTORS_OPERATIONS_TOPO_VIEW_HH

#include "flecsolve/util/future.hh"
#include "flecsolve/util/traits.hh"
#include "flecsolve/vectors/data/topo_view.hh"
#include "topo_tasks.hh"

namespace flecsolve::vec::ops {

template<class Topo, typename Topo::index_space Space, class Scalar>
struct topo_view {
	using scalar = Scalar;
	using real = typename num_traits<Scalar>::real;
	using len_t = flecsi::util::id;

	using vec_data = data::topo_view<Topo, Space, scalar>;

	using tasks = topo_tasks<vec_data, scalar, len_t>;

	template<class Other>
	void copy(const Other & x, vec_data & z) {
		static_assert(
			std::is_same_v<typename Other::topo_t, typename vec_data::topo_t>);
		static_assert(Other::space == vec_data::space);
		flecsi::execute<
			tasks::template copy<typename Other::template acc_all<flecsi::ro>>,
			flecsi::default_accelerator>(x.topo(), z.ref(), x.ref());
	}

	void zero(vec_data & x) {
		flecsi::execute<tasks::set_to_scalar, flecsi::default_accelerator>(
			x.topo(), x.ref(), 0.0);
	}

	void set_random(vec_data & x, unsigned seed) {
		flecsi::execute<tasks::set_random>(x.topo(), x.ref(), seed);
	}

	void set_to_scalar(scalar alpha, vec_data & x) {
		flecsi::execute<tasks::set_to_scalar, flecsi::default_accelerator>(
			x.topo(), x.ref(), alpha);
	}

	void scale(scalar alpha, vec_data & x) {
		flecsi::execute<tasks::scale_self, flecsi::default_accelerator>(
			x.topo(), x.ref(), alpha);
	}

	void scale(scalar alpha, const vec_data & x, vec_data & y) {
		flog_assert(x.fid() != y.fid(),
		            "scale operation: vector data cannot be the same");
		flecsi::execute<tasks::scale, flecsi::default_accelerator>(
			x.topo(), x.ref(), y.ref(), alpha);
	}

	void add(const vec_data & x, const vec_data & y, vec_data & z) {
		if (x.fid() == z.fid()) {
			flecsi::execute<tasks::add_self, flecsi::default_accelerator>(
				z.topo(), z.ref(), y.ref());
		}
		else if (y.fid() == z.fid()) {
			flecsi::execute<tasks::add_self, flecsi::default_accelerator>(
				z.topo(), z.ref(), x.ref());
		}
		else {
			flecsi::execute<tasks::add, flecsi::default_accelerator>(
				z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	void subtract(const vec_data & x, const vec_data & y, vec_data & z) {
		if (x.fid() == z.fid()) {
			flecsi::execute<tasks::template subtract_self<true>,
			                flecsi::default_accelerator>(
				z.topo(), z.ref(), y.ref());
		}
		else if (y.fid() == z.fid()) {
			flecsi::execute<tasks::template subtract_self<false>,
			                flecsi::default_accelerator>(
				z.topo(), z.ref(), x.ref());
		}
		else {
			flecsi::execute<tasks::subtract, flecsi::default_accelerator>(
				z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	void multiply(const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			flecsi::execute<tasks::multiply_self, flecsi::default_accelerator>(
				z.topo(), z.ref(), y.ref());
		}
		else if (z.fid() == y.fid()) {
			flecsi::execute<tasks::multiply_self, flecsi::default_accelerator>(
				z.topo(), z.ref(), x.ref());
		}
		else {
			flecsi::execute<tasks::multiply, flecsi::default_accelerator>(
				z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	void divide(const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			flecsi::execute<tasks::template divide_self<true>,
			                flecsi::default_accelerator>(
				z.topo(), z.ref(), y.ref());
		}
		else if (z.fid() == y.fid()) {
			flecsi::execute<tasks::template divide_self<false>,
			                flecsi::default_accelerator>(
				z.topo(), z.ref(), x.ref());
		}
		else {
			flecsi::execute<tasks::divide, flecsi::default_accelerator>(
				z.topo(), z.ref(), x.ref(), y.ref());
		}
	}

	void reciprocal(const vec_data & x, vec_data & y) {
		if (x.fid() == y.fid()) {
			flecsi::execute<tasks::reciprocal_self,
			                flecsi::default_accelerator>(y.topo(), y.ref());
		}
		else {
			flecsi::execute<tasks::reciprocal, flecsi::default_accelerator>(
				y.topo(), y.ref(), x.ref());
		}
	}

	void linear_sum(scalar alpha,
	                const vec_data & x,
	                scalar beta,
	                const vec_data & y,
	                vec_data & z) {
		if (z.fid() == x.fid()) {
			flecsi::execute<tasks::template linear_sum_self<true>,
			                flecsi::default_accelerator>(
				z.topo(), z.ref(), y.ref(), alpha, beta);
		}
		else if (z.fid() == y.fid()) {
			flecsi::execute<tasks::template linear_sum_self<false>,
			                flecsi::default_accelerator>(
				z.topo(), x.ref(), z.ref(), alpha, beta);
		}
		else {
			flecsi::execute<tasks::linear_sum, flecsi::default_accelerator>(
				z.topo(), z.ref(), alpha, x.ref(), beta, y.ref());
		}
	}

	void
	axpy(scalar alpha, const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			flecsi::execute<tasks::template axpy_self<true>,
			                flecsi::default_accelerator>(
				z.topo(), z.ref(), y.ref(), alpha);
		}
		else if (z.fid() == y.fid()) {
			flecsi::execute<tasks::template axpy_self<false>,
			                flecsi::default_accelerator>(
				z.topo(), z.ref(), x.ref(), alpha);
		}
		else {
			flecsi::execute<tasks::axpy, flecsi::default_accelerator>(
				z.topo(), z.ref(), alpha, x.ref(), y.ref());
		}
	}

	void axpby(scalar alpha, scalar beta, const vec_data & x, vec_data & z) {
		flecsi::execute<tasks::axpby, flecsi::default_accelerator>(
			z.topo(), z.ref(), x.ref(), alpha, beta);
	}

	void abs(const vec_data & x, vec_data & y) {
		if (y.fid() == x.fid()) {
			flecsi::execute<tasks::abs_self, flecsi::default_accelerator>(
				y.topo(), y.ref());
		}
		else {
			flecsi::execute<tasks::abs, flecsi::default_accelerator>(
				y.topo(), y.ref(), x.ref());
		}
	}

	void add_scalar(const vec_data & x, scalar alpha, vec_data & y) {
		if (x.fid() == y.fid()) {
			flecsi::execute<tasks::add_scalar_self,
			                flecsi::default_accelerator>(
				y.topo(), y.ref(), alpha);
		}
		else {
			flecsi::execute<tasks::add_scalar, flecsi::default_accelerator>(
				y.topo(), y.ref(), x.ref(), alpha);
		}
	}

	auto min(const vec_data & x) const {
		return flecsi::reduce<tasks::local_min,
		                      flecsi::exec::fold::min,
		                      flecsi::default_accelerator>(x.topo(), x.ref());
	}

	auto max(const vec_data & y) const {
		return flecsi::reduce<tasks::local_max,
		                      flecsi::exec::fold::max,
		                      flecsi::default_accelerator>(y.topo(), y.ref());
	}

	template<unsigned short p>
	auto lp_norm_local(const vec_data & x) const {
		if constexpr (p == 1) {
			return flecsi::reduce<tasks::l1_norm_local,
			                      flecsi::exec::fold::sum,
			                      flecsi::default_accelerator>(x.topo(),
			                                                   x.ref());
		}
		else if constexpr (p == 2) {
			return flecsi::reduce<tasks::l2_norm_local,
			                      flecsi::exec::fold::sum,
			                      flecsi::default_accelerator>(x.topo(),
			                                                   x.ref());
		}
		else {
			return flecsi::reduce<tasks::lp_norm_local,
			                      flecsi::exec::fold::sum,
			                      flecsi::default_accelerator>(
					   x.topo(), x.ref(), p)
			    .get();
		}
	}

	template<unsigned short p>
	auto lp_norm(const vec_data & x) const {
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

	auto inf_norm(const vec_data & x) const {
		return flecsi::reduce<tasks::inf_norm_local,
		                      flecsi::exec::fold::max,
		                      flecsi::default_accelerator>(x.topo(), x.ref());
	}

	auto dot(const vec_data & x, const vec_data & y) const {
		return flecsi::reduce<tasks::scalar_prod,
		                      flecsi::exec::fold::sum,
		                      flecsi::default_accelerator>(
			x.topo(), x.ref(), y.ref());
	}

	auto global_size(const vec_data & x) const {
		return flecsi::reduce<tasks::local_size,
		                      flecsi::exec::fold::sum,
		                      flecsi::default_accelerator>(x.topo());
	}

	len_t local_size(const vec_data & x) const {
		len_t length;
		flecsi::execute<tasks::get_local_size, flecsi::mpi>(x.topo(), &length);
		return length;
	}

	void dump(std::string_view pre, const vec_data & x) const {
		// TODO: update for multiaccessor
		flecsi::execute<tasks::dump, flecsi::mpi>(pre, x.topo(), x.ref());
	}

	template<class F, class... Vecs>
	static constexpr decltype(auto) apply(F && f, Vecs &&... vecs) {
		return std::forward<F>(f)(std::forward<Vecs>(vecs)...);
	}
};

}

#endif
