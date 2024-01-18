#ifndef FLECSI_LINALG_VECTORS_OPS_MULTI_H
#define FLECSI_LINALG_VECTORS_OPS_MULTI_H

#include <tuple>
#include <functional>
#include <cmath>

#include "flecsolve/util/future.hh"

namespace flecsolve::vec::ops {

template<class Data>
struct multi {
	using config = typename Data::config;
	static constexpr std::size_t num_vecs = config::num_components;
	using scalar = typename config::scalar;
	using real = typename config::real;
	using vec_data = Data;

	template<class T>
	static void copy(const T & x, vec_data & z) {
		apply([](auto & x, const auto & y) { x.copy(y); },
		      make_is(),
		      z.components,
		      x.components);
	}

	static void zero(vec_data & x) {
		apply([](auto & x) { x.zero(); }, make_is(), x.components);
	}

	static void set_to_scalar(scalar alpha, vec_data & x) {
		apply([alpha](auto & x) { x.set_scalar(alpha); },
		      make_is(),
		      x.components);
	}

	static void scale(scalar alpha, vec_data & x) {
		apply([alpha](auto & x) { x.scale(alpha); }, make_is(), x.components);
	}

	template<class T>
	static void scale(scalar alpha, const T & x, vec_data & y) {
		apply([alpha](auto & v, const auto & y) { v.scale(alpha, y); },
		      make_is(),
		      y.components,
		      x.components);
	}

	template<class T0, class T1>
	static void add(const T0 & x, const T1 & y, vec_data & z) {
		apply([](auto & x, const auto & y, const auto & z) { x.add(y, z); },
		      make_is(),
		      z.components,
		      x.components,
		      y.components);
	}

	template<class T0, class T1>
	static void subtract(const T0 & x, const T1 & y, vec_data & z) {
		apply(
			[](auto & x, const auto & y, const auto & z) { x.subtract(y, z); },
			make_is(),
			z.components,
			x.components,
			y.components);
	}

	template<class T0, class T1>
	static void multiply(const T0 & x, const T1 & y, vec_data & z) {
		apply(
			[](auto & x, const auto & y, const auto & z) { x.multiply(y, z); },
			make_is(),
			z.components,
			x.components,
			y.components);
	}

	template<class T0, class T1>
	static void divide(const T0 & x, const T1 & y, vec_data & z) {
		apply([](auto & x, const auto & y, const auto & z) { x.divide(y, z); },
		      make_is(),
		      z.components,
		      x.components,
		      y.components);
	}

	template<class T>
	static void reciprocal(const T & x, vec_data & y) {
		apply([](auto & x, const auto & y) { x.reciprocal(y); },
		      make_is(),
		      y.components,
		      x.components);
	}

	template<class T0, class T1>
	static void linear_sum(scalar alpha,
	                       const T0 & x,
	                       scalar beta,
	                       const T1 & y,
	                       vec_data & z) {
		apply(
			[alpha, beta](auto & z, const auto & x, const auto & y) {
				z.linear_sum(alpha, x, beta, y);
			},
			make_is(),
			z.components,
			x.components,
			y.components);
	}

	template<class T0, class T1>
	static void axpy(scalar alpha, const T0 & x, const T1 & y, vec_data & z) {
		apply([alpha](auto & z,
		              const auto & x,
		              const auto & y) { z.axpy(alpha, x, y); },
		      make_is(),
		      z.components,
		      x.components,
		      y.components);
	}

	template<class T>
	static void axpby(scalar alpha, scalar beta, const T & x, vec_data & z) {
		apply([alpha, beta](auto & z,
		                    const auto & x) { z.axpby(alpha, beta, x); },
		      make_is(),
		      z.components,
		      x.components);
	}

	template<class T>
	static void abs(const T & x, vec_data & y) {
		apply([](auto & z, const auto & y) { z.abs(y); },
		      make_is(),
		      y.components,
		      x.components);
	}

	template<class T>
	static void add_scalar(const T & x, scalar alpha, vec_data & y) {
		apply([alpha](auto & z, const auto & x) { z.add_scalar(x, alpha); },
		      make_is(),
		      y.components,
		      x.components);
	}

	static void set_random(vec_data & x, unsigned seed) {
		apply([=](auto & z) { z.set_random(seed); }, make_is(), x.components);
	}

	static auto min(const vec_data & x) {
		auto futs = apply_ret(
			[](const auto & x) { return x.min(); }, make_is(), x.components);

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

	static auto max(const vec_data & y) {
		auto futs = apply_ret(
			[](const auto & x) { return x.max(); }, make_is(), y.components);

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
	static auto lp_norm(const vec_data & x) {
		auto futs = apply_ret(
			[](const auto & x) {
				return std::remove_reference_t<
					decltype(x)>::ops::template lp_norm_local<p>(x.data);
			},
			make_is(),
			x.components);

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

	static auto inf_norm(const vec_data & x) {
		auto futs = apply_ret([](const auto & x) { return x.inf_norm(); },
		                      make_is(),
		                      x.components);

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
	static auto dot(const vec_data & x, const T & y) {
		auto futs =
			apply_ret([](const auto & x, const auto & y) { return x.dot(y); },
		              make_is(),
		              x.components,
		              y.components);

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
