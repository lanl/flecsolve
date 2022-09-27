#ifndef FLECSI_LINALG_VECTORS_OPS_MULTI_H
#define FLECSI_LINALG_VECTORS_OPS_MULTI_H

#include <tuple>
#include <functional>

#include "flecsolve/util/future.hh"
#include "flecsolve/util/traits.hh"

namespace flecsolve::vec::ops {

template<class Scalar, class Len, class VecData, std::size_t NumVecs>
struct multi {
	static constexpr std::size_t num_vecs = NumVecs;
	using vec_data = VecData;
	using scalar = Scalar;
	using real = typename num_traits<scalar>::real;
	using len_t = Len;

	template<class T>
	void copy(const T & x, vec_data & z) {
		apply([](auto & x, const auto & y) { x.copy(y); }, make_is(), z, x);
	}

	void zero(vec_data & x) {
		apply([](auto & x) { x.zero(); }, make_is(), x);
	}

	void set_to_scalar(scalar alpha, vec_data & x) {
		apply([alpha](auto & x) { x.set_scalar(alpha); }, make_is(), x);
	}

	void scale(scalar alpha, vec_data & x) {
		apply([alpha](auto & x) { x.scale(alpha); }, make_is(), x);
	}

	template<class T>
	void scale(scalar alpha, const T & x, vec_data & y) {
		apply([alpha](auto & v, const auto & y) { v.scale(alpha, y); },
		      make_is(),
		      y,
		      x);
	}

	template<class T0, class T1>
	void add(const T0 & x, const T1 & y, vec_data & z) {
		apply([](auto & x, const auto & y, const auto & z) { x.add(y, z); },
		      make_is(),
		      z,
		      x,
		      y);
	}

	template<class T0, class T1>
	void subtract(const T0 & x, const T1 & y, vec_data & z) {
		apply(
			[](auto & x, const auto & y, const auto & z) { x.subtract(y, z); },
			make_is(),
			z,
			x,
			y);
	}

	template<class T0, class T1>
	void multiply(const T0 & x, const T1 & y, vec_data & z) {
		apply(
			[](auto & x, const auto & y, const auto & z) { x.multiply(y, z); },
			make_is(),
			z,
			x,
			y);
	}

	template<class T0, class T1>
	void divide(const T0 & x, const T1 & y, vec_data & z) {
		apply([](auto & x, const auto & y, const auto & z) { x.divide(y, z); },
		      make_is(),
		      z,
		      x,
		      y);
	}

	template<class T>
	void reciprocal(const T & x, vec_data & y) {
		apply(
			[](auto & x, const auto & y) { x.reciprocal(y); }, make_is(), y, x);
	}

	template<class T0, class T1>
	void linear_sum(scalar alpha,
	                const T0 & x,
	                scalar beta,
	                const T1 & y,
	                vec_data & z) {
		apply(
			[alpha, beta](auto & z, const auto & x, const auto & y) {
				z.linear_sum(alpha, x, beta, y);
			},
			make_is(),
			z,
			x,
			y);
	}

	template<class T0, class T1>
	void axpy(scalar alpha, const T0 & x, const T1 & y, vec_data & z) {
		apply([alpha](auto & z,
		              const auto & x,
		              const auto & y) { z.axpy(alpha, x, y); },
		      make_is(),
		      z,
		      x,
		      y);
	}

	template<class T>
	void axpby(scalar alpha, scalar beta, const T & x, vec_data & z) {
		apply([alpha, beta](auto & z,
		                    const auto & x) { z.axpby(alpha, beta, x); },
		      make_is(),
		      z,
		      x);
	}

	template<class T>
	void abs(const T & x, vec_data & y) {
		apply([](auto & z, const auto & y) { z.abs(y); }, make_is(), y, x);
	}

	template<class T>
	void add_scalar(const T & x, scalar alpha, vec_data & y) {
		apply([alpha](auto & z, const auto & x) { z.add_scalar(x, alpha); },
		      make_is(),
		      y,
		      x);
	}

	void set_random(vec_data & x, unsigned seed) {
		apply([=](auto & z) { z.set_random(seed); }, make_is(), x);
	}

	auto min(const vec_data & x) const {
		auto futs =
			apply_ret([](const auto & x) { return x.min(); }, make_is(), x);

		return future_transform{
			future_vector{std::move(futs)}, [](auto && v) {
				return std::apply(
					[](auto... vs) {
						std::array<real, num_vecs> vals{vs...};
						return *std::min_element(vals.begin(), vals.end());
					},
					v);
			}};
	}

	auto max(const vec_data & y) const {
		auto futs =
			apply_ret([](const auto & x) { return x.max(); }, make_is(), y);

		return future_transform{
			future_vector{std::move(futs)}, [](auto && v) {
				return std::apply(
					[](auto... vs) {
						std::array<real, num_vecs> vals{vs...};
						return *std::max_element(vals.begin(), vals.end());
					},
					v);
			}};
	}

	template<unsigned short p>
	auto lp_norm(const vec_data & x) const {
		auto futs = apply_ret(
			[](const auto & x) {
				return x.ops.template lp_norm_local<p>(x.data);
			},
			make_is(),
			x);

		return future_transform{future_vector{std::move(futs)}, [](auto && v) {
									return std::apply(
										[](auto... vs) {
											auto sum = (vs + ...);
											if constexpr (p == 1) {
												return sum;
											}
											else if constexpr (p == 2) {
												return std::sqrt(sum);
											}
											else {
												return std::pow(sum, 1. / p);
											}
										},
										v);
								}};
	}

	auto inf_norm(const vec_data & x) const {
		auto futs = apply_ret(
			[](const auto & x) { return x.inf_norm(); }, make_is(), x);

		return future_transform{
			future_vector{std::move(futs)}, [](auto && v) {
				return std::apply(
					[](auto... vs) {
						std::array<real, num_vecs> vals{vs...};
						return *std::max_element(vals.begin(), vals.end());
					},
					v);
			}};
	}

	template<class T>
	auto dot(const vec_data & x, const T & y) const {
		auto futs =
			apply_ret([](const auto & x, const auto & y) { return x.dot(y); },
		              make_is(),
		              x,
		              y);

		return future_transform{
			future_vector{std::move(futs)}, [](auto && v) {
				return std::apply([](auto... vs) { return (vs + ...); }, v);
			}};
	}

	template<std::size_t I, class F, class... Multis>
	static constexpr decltype(auto) apply_aux(F && f, Multis &&... ms) {
		return std::invoke(std::forward<F>(f),
		                   std::get<I>(std::forward<Multis>(ms))...);
	}

	template<class F, std::size_t... Index, class... Multis>
	static constexpr void
	apply(F && f, std::index_sequence<Index...>, Multis &&... ms) {
		(apply_aux<Index>(std::forward<F>(f), std::forward<Multis>(ms)...),
		 ...);
	}

	template<class F, std::size_t... Index, class... Multis>
	static constexpr decltype(auto)
	apply_ret(F && f, std::index_sequence<Index...>, Multis &&... ms) {
		return std::make_tuple(apply_aux<Index>(
			std::forward<F>(f), std::forward<Multis>(ms)...)...);
	}

protected:
	using make_is = std::make_index_sequence<num_vecs>;
};

}
#endif
