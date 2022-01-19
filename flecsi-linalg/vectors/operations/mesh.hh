#pragma once

#include "flecsi-linalg/util/future.hh"
#include "mesh_tasks.hh"

namespace flecsi::linalg::vec::ops {

template<class Topo, typename Topo::index_space Space, class VecTypes>
struct mesh {
	using scalar = typename VecTypes::scalar;
	using real = typename VecTypes::real;
	using len_t = typename VecTypes::len;

	using vec_data = data::mesh<Topo, Space, scalar>;

	using tasks = mesh_tasks<vec_data, VecTypes>;

	void copy(const vec_data & x, vec_data & z) {
		execute<tasks::copy>(x.topo, z.ref(), x.ref());
	}

	void zero(vec_data & x) {
		execute<tasks::set_to_scalar>(x.topo, x.ref(), 0.0);
	}

	void set_random(vec_data & x) {
		execute<tasks::set_random>(x.topo, x.ref());
	}

	void set_to_scalar(scalar alpha, vec_data & x) {
		execute<tasks::set_to_scalar>(x.topo, x.ref(), alpha);
	}

	void scale(scalar alpha, vec_data & x) {
		execute<tasks::scale_self>(x.topo, x.ref(), alpha);
	}

	void scale(scalar alpha,
	           const vec_data & x,
	           vec_data & y) {
		flog_assert(x.fid() != y.fid(), "scale operation: vector data cannot be the same");
		execute<tasks::scale>(x.topo, x.ref(), y.ref(), alpha);
	}

	void add(const vec_data & x, const vec_data & y, vec_data & z) {
		if (x.fid() == z.fid()) {
			execute<tasks::add_self>(z.topo, z.ref(), y.ref());
		} else if (y.fid() == z.fid()) {
			execute<tasks::add_self>(z.topo, z.ref(), x.ref());
		} else {
			execute<tasks::add>(z.topo, z.ref(), x.ref(), y.ref());
		}
	}

	void subtract(const vec_data & x, const vec_data & y, vec_data & z) {
		if (x.fid() == z.fid()) {
			execute<tasks::template subtract_self<true>>(z.topo, z.ref(), y.ref());
		} else if (y.fid() == z.fid()) {
			execute<tasks::template subtract_self<false>>(z.topo, z.ref(), x.ref());
		} else {
			execute<tasks::subtract>(z.topo, z.ref(), x.ref(), y.ref());
		}
	}

	void multiply(const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			execute<tasks::multiply_self>(z.topo, z.ref(), y.ref());
		} else if (z.fid() == y.fid()) {
			execute<tasks::multiply_self>(z.topo, z.ref(), x.ref());
		} else {
			execute<tasks::multiply>(z.topo, z.ref(), x.ref(), y.ref());
		}
	}

	void divide(const vec_data & x, const vec_data & y, vec_data & z) {
		if (z.fid() == x.fid()) {
			execute<tasks::template divide_self<true>>(z.topo, z.ref(), y.ref());
		} else if (z.fid() == y.fid()) {
			execute<tasks::template divide_self<false>>(z.topo, z.ref(), x.ref());
		} else {
			execute<tasks::divide>(z.topo, z.ref(), x.ref(), y.ref());
		}
	}

	void reciprocal(const vec_data & x, vec_data & y) {
		if (x.fid() == y.fid()) {
			execute<tasks::reciprocal_self>(y.topo, y.ref());
		} else {
			execute<tasks::reciprocal>(y.topo, y.ref(), x.ref());
		}
	}

	void linear_sum(scalar alpha, const vec_data & x,
	                scalar beta, const vec_data & y,
	                vec_data & z) {
		if (z.fid() == x.fid()) {
			execute<tasks::template linear_sum_self<true>>(z.topo, z.ref(), y.ref(),
			                                               alpha, beta);
		} else if (z.fid() == y.fid()) {
			execute<tasks::template linear_sum_self<false>>(z.topo, x.ref(), z.ref(),
			                                                alpha, beta);
		} else {
			execute<tasks::linear_sum>(z.topo, z.ref(), alpha, x.ref(),
			                           beta, y.ref());
		}
	}

	void axpy(scalar alpha,
	          const vec_data & x, const vec_data & y,
	          vec_data & z) {
		if (z.fid() == x.fid()) {
			execute<tasks::template axpy_self<true>>(z.topo, z.ref(), y.ref(), alpha);
		} else if (z.fid() == y.fid()) {
			execute<tasks::template axpy_self<false>>(z.topo, z.ref(), x.ref(), alpha);
		} else {
			execute<tasks::axpy>(z.topo, z.ref(), alpha, x.ref(), y.ref());
		}
	}

	void axpby(scalar alpha, scalar beta,
	           const vec_data & x,
	           vec_data & z) {
		execute<tasks::axpby>(z.topo, z.ref(), x.ref(),
		                      alpha, beta);
	}

	void abs(const vec_data & x, vec_data & y) {
		if (y.fid() == x.fid()) {
			execute<tasks::abs_self>(y.topo, y.ref());
		} else {
			execute<tasks::abs>(y.topo, y.ref(), x.ref());
		}
	}

	void add_scalar(const vec_data & x,
	                scalar alpha,
	                vec_data & y) {
		if (x.fid() == y.fid()) {
			execute<tasks::add_scalar_self>(y.topo, y.ref(), alpha);
		} else {
			execute<tasks::add_scalar>(y.topo, y.ref(), x.ref(), alpha);
		}
	}

	auto min(const vec_data & x) const {
		return reduce<tasks::local_min,
			flecsi::exec::fold::min>(x.topo, x.ref());
	}

	auto max(const vec_data & y) const {
		return reduce<tasks::local_max,
		              flecsi::exec::fold::max>(y.topo, y.ref());
	}

	template<unsigned short p>
	auto lp_norm_local(const vec_data & x) const {
		if constexpr (p == 1) {
			return reduce<tasks::l1_norm_local,
			              exec::fold::sum>(x.topo, x.ref());
		} else if constexpr (p == 2) {
			return reduce<tasks::l2_norm_local,
			              exec::fold::sum>(x.topo, x.ref());
		} else {
			return reduce<tasks::lp_norm_local,
			              exec::fold::sum>(x.topo, x.ref(), p).get();
		}
	}

	template<unsigned short p>
	auto lp_norm(const vec_data & x) const {
		auto fut = lp_norm_local<p>(x);
		if constexpr (p == 1) {
			return fut;
		} else if constexpr (p == 2) {
			return future_transform{std::move(fut), [](auto v) {
				return std::sqrt(v);}};
		} else {
			return future_transform{std::move(fut), [](auto v) {
				return std::pow(v, 1./p);
			}};
		}
	}

	auto inf_norm(const vec_data & x) const {
		return reduce<tasks::inf_norm_local, exec::fold::max>(x.topo, x.ref());
	}

	auto inner_prod(const vec_data & x, const vec_data & y) const {
		return reduce<tasks::scalar_prod,
			exec::fold::sum>(x.topo, x.ref(), y.ref());
	}

	auto global_size(const vec_data & x) const {
		return reduce<tasks::local_size, exec::fold::sum>(x.topo);
	}

	len_t local_size(const vec_data & x) const {
		len_t length;
		execute<tasks::get_local_size, mpi>(x.topo, &length);
		return length;
	}
};

}
