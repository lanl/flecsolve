#pragma once

#include <tuple>
#include <functional>

#include "flecsi-linalg/util/future.hh"

namespace flecsi::linalg::vec::ops {

template <class VecTypes, class... Vecs>
struct multi {
	using vec_data = std::tuple<Vecs...>;
	using scalar = typename VecTypes::scalar;
	using real = typename VecTypes::real;
	using len_t = typename VecTypes::len;

	void copy(const vec_data & x, vec_data & z) {
		apply([](auto & x, const auto & y) {
			x.copy(y);
		}, make_is(), z, x);
	}

	void zero(vec_data & x) {
		apply([](auto & x) {
			x.zero();
		}, make_is(), x);
	}

	void set_to_scalar(scalar alpha, vec_data & x) {
		apply([alpha](auto & x) {
			x.set_to_scalar(alpha);
		}, make_is(), x);
	}

	void scale(scalar alpha, vec_data & x) {
		apply([alpha](auto & x) {
			x.scale(alpha);
		}, make_is(), x);
	}

	void scale(scalar alpha,
	           const vec_data & x,
	           vec_data & y) {
		apply([alpha](auto & v, const auto & y) {
			v.scale(alpha, y);
		}, make_is(), y, x);
	}

	void add(const vec_data & x, const vec_data & y, vec_data & z) {
		apply([](auto & x, const auto & y, const auto & z) {
			x.add(y, z);
		}, make_is(), z, x, y);
	}

	void subtract(const vec_data & x, const vec_data & y, vec_data & z) {
		apply([](auto & x, const auto & y, const auto & z) {
			x.subtract(y, z);
		}, make_is(), z, x, y);
	}

	void multiply(const vec_data & x, const vec_data & y, vec_data & z) {
		apply([](auto & x, const auto & y, const auto & z) {
			x.multiply(y, z);
		}, make_is(), z, x, y);
	}

	void divide(const vec_data & x, const vec_data & y, vec_data & z) {
		apply([](auto & x, const auto & y, const auto & z) {
			x.divide(y, z);
		}, make_is(), z, x, y);
	}

	void reciprocal(const vec_data & x, vec_data & y) {
		apply([](auto & x, const auto & y) {
			x.reciprocal(y);
		}, make_is(), y, x);
	}

	void linear_sum(scalar alpha, const vec_data & x,
	                scalar beta, const vec_data & y,
	                vec_data & z) {
		apply([alpha,beta](auto & z, const auto & x, const auto & y) {
			z.linear_sum(alpha, x, beta, y);
		}, make_is(), z, x, y);
	}

	void axpy(scalar alpha,
	          const vec_data & x, const vec_data & y,
	          vec_data & z) {
		apply([alpha](auto & z, const auto & x, const auto & y) {
			z.axpy(alpha, x, y);
		}, make_is(), z, x, y);
	}

	void axpby(scalar alpha, scalar beta,
	           const vec_data & x,
	           vec_data & z) {
		apply([alpha, beta](auto & z, const auto & x) {
			z.axpby(alpha, beta, x);
		}, make_is(), z, x);
	}

	void abs(const vec_data & x, vec_data & y) {
		apply([](auto & z, const auto & y) {
			z.abs(y);
		}, make_is(), y, x);
	}

	void add_scalar(const vec_data & x,
	                scalar alpha,
	                vec_data & y) {
		apply([alpha](auto & z, const auto & x) {
			z.add_scalar(x, alpha);
		}, make_is(), y, x);
	}

	auto min(const vec_data & x) const {
		auto futs = apply_ret([](const auto & x) {
			return x.min();
		}, make_is(), x);

		return future_transform{
			future_vector{std::move(futs)},
			[](auto && v) {
				return std::apply([](auto ...vs) {
					std::array<real, sizeof...(Vecs)> vals{
						vs...};
					return *std::min_element(vals.begin(), vals.end());
				}, v);
			}};
	}

	auto max(const vec_data & y) const {
		auto futs = apply_ret([](const auto & x) {
			return x.max();
		}, make_is(), y);

		return future_transform{
			future_vector{std::move(futs)},
			[](auto && v) {
				return std::apply([](auto ...vs) {
					std::array<real, sizeof...(Vecs)> vals{
						vs...};
					return *std::max_element(vals.begin(), vals.end());
				}, v);
			}};
	}


	template<unsigned short p>
	auto lp_norm(const vec_data & x) const {
		auto futs = apply_ret([](const auto & x) {
			return x.ops.template
				lp_norm_local<p>(x.data);
		}, make_is(), x);

		return future_transform{
			future_vector{std::move(futs)},
			[](auto && v) {
				return std::apply([](auto ...vs) {
					auto sum = (vs + ...);
					if constexpr (p == 1) {
						return sum;
					} else if constexpr (p == 2) {
						return std::sqrt(sum);
					} else {
						return std::pow(sum, 1./p);
					}
				}, v);
			}};
	}

	auto inf_norm(const vec_data & x) const {
		auto futs = apply_ret([](const auto & x) {
			return x.inf_norm();
		}, make_is(), x);

		return future_transform{
			future_vector{std::move(futs)},
			[](auto && v) {
				return std::apply([](auto ...vs) {
					std::array<real, sizeof...(Vecs)> vals{
						vs...};
					return *std::max_element(vals.begin(), vals.end());
				}, v);
			}};
	}

	auto inner_prod(const vec_data & x, const vec_data & y) const {
		auto futs = apply_ret([](const auto & x, const auto & y) {
			return x.inner_prod(y);
		}, make_is(), x, y);

		return future_transform{
			future_vector{std::move(futs)},
			[](auto && v) {
				return std::apply([](auto ...vs) {
					return (vs + ...);
				}, v);
			}};
	}

protected:
	template<std::size_t I, class F, class ... Multis>
	constexpr decltype(auto) apply_aux(F && f, Multis&& ... ms) const {
		return std::invoke(std::forward<F>(f),
		                   std::get<I>(std::forward<Multis>(ms))...);
	}

	template<class F, std::size_t ... Index, class ... Multis>
	constexpr void apply(F && f, std::index_sequence<Index...>,
	                     Multis&& ... ms) const {
		(apply_aux<Index>(std::forward<F>(f), std::forward<Multis>(ms)...), ...);
	}

	template<class F, std::size_t ... Index, class ... Multis>
	constexpr decltype(auto) apply_ret(F && f, std::index_sequence<Index...>,
	                     Multis&& ... ms) const {
		return std::make_tuple(apply_aux<Index>(std::forward<F>(f), std::forward<Multis>(ms)...)...);
	}

	using make_is = std::make_index_sequence<sizeof...(Vecs)>;
};

}
