#pragma once

#include <tuple>
#include <functional>

#include "flecsi-linalg/util/future.hh"
#include "operations/flecsi_operations.hh"

namespace flecsi::linalg {

template <class... Vecs>
struct multivector
{
	using vec = multivector<Vecs...>;
	using real_t = typename std::tuple_element<0, std::tuple<Vecs...>>::type::real_t;
	multivector(Vecs... vs) :
		vecs{std::forward<Vecs>(vs)...}
	{
	}

	template<std::size_t I>
	constexpr auto & get() & {
		return std::get<I>(vecs);
	}

	template<std::size_t I>
	constexpr const auto & get() const & {
		return std::get<I>(vecs);
	}

	void copy(const vec & other) {
		apply([](auto & x, const auto & y) {
			x.copy(y);
		}, make_is(), *this, other);
	}

	void zero() {
		apply([](auto & x) {
			x.zero();
		}, make_is(), *this);
	}

	void set_to_scalar(real_t val) {
		apply([val](auto & x) {
			x.set_to_scalar(val);
		}, make_is(), *this);
	}

	void scale(real_t alpha, const vec & x) {
		apply([alpha](auto & v, const auto & y) {
			v.scale(alpha, y);
		}, make_is(), *this, x);
	}

	void scale(real_t alpha) {
		apply([alpha](auto & x) {
			x.scale(alpha);
		}, make_is(), *this);
	}

	void add(const vec & x, const vec & y) {
		apply([](auto & x, const auto & y, const auto & z) {
			x.add(y, z);
		}, make_is(), *this, x, y);
	}

	void subtract(const vec & x, const vec & y) {
		apply([](auto & x, const auto & y, const auto & z) {
			x.subtract(y, z);
		}, make_is(), *this, x, y);
	}

	void multiply(const vec & x, const vec & y) {
		apply([](auto & x, const auto & y, const auto & z) {
			x.multiply(y, z);
		}, make_is(), *this, x, y);
	}

	void divide(const vec & x, const vec & y) {
		apply([](auto & x, const auto & y, const auto & z) {
			x.divide(y, z);
		}, make_is(), *this, x, y);
	}

	void reciprocal(const vec & x) {
		apply([](auto & x, const auto & y) {
			x.reciprocal(y);
		}, make_is(), *this, x);
	}

	void linear_sum(real_t alpha, const vec & x, real_t beta,
	                const vec & y) {
		apply([alpha,beta](auto & z, const auto & x, const auto & y) {
			z.linear_sum(alpha, x, beta, y);
		}, make_is(), *this, x, y);
	}

	void axpy(real_t alpha, const vec & x, const vec & y) {
		apply([alpha](auto & z, const auto & x, const auto & y) {
			z.axpy(alpha, x, y);
		}, make_is(), *this, x, y);
	}

	void axpby(real_t alpha, real_t beta, const vec & x) {
		apply([alpha, beta](auto & z, const auto & x) {
			z.axpby(alpha, beta, x);
		}, make_is(), *this, x);
	}

	void abs(const vec & y) {
		apply([](auto & z, const auto & y) {
			z.abs(y);
		}, make_is(), *this, y);
	}

	void add_scalar(const vec & x, real_t alpha) {
		apply([alpha](auto & z, const auto & x) {
			z.add_scalar(x, alpha);
		}, make_is(), *this, x);
	}

	auto min() const {
		auto futs = apply_ret([](const auto & x) {
			return x.min();
		}, make_is(), *this);

		return future_transform{
			future_vector{std::move(futs)},
			[](auto && v) {
				return std::apply([](auto ...vs) {
					std::array<real_t, sizeof...(Vecs)> vals{
						vs...};
					return *std::min_element(vals.begin(), vals.end());
				}, v);
			}};
	}

	auto max() const {
		auto futs = apply_ret([](const auto & x) {
			return x.max();
		}, make_is(), *this);

		return future_transform{
			future_vector{std::move(futs)},
			[](auto && v) {
				return std::apply([](auto ...vs) {
					std::array<real_t, sizeof...(Vecs)> vals{
						vs...};
					return *std::max_element(vals.begin(), vals.end());
				}, v);
			}};
	}

	template<unsigned short p>
	auto lp_norm() const {
		auto futs = apply_ret([](const auto & x) {
			return x.ops.template
				lp_norm_local<p>(x.data);
		}, make_is(), *this);

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

	auto l1norm() const {
		return lp_norm<1>();
	}

	auto l2norm() const {
		return lp_norm<2>();
	}

	auto inf_norm() const {
		auto futs = apply_ret([](const auto & x) {
			return x.inf_norm();
		}, make_is(), *this);

		return future_transform{
			future_vector{std::move(futs)},
			[](auto && v) {
				return std::apply([](auto ...vs) {
					std::array<real_t, sizeof...(Vecs)> vals{
						vs...};
					return *std::max_element(vals.begin(), vals.end());
				}, v);
			}};
	}

	auto inner_prod(const vec & x) const {
		auto futs = apply_ret([](const auto & x, const auto & y) {
			return x.inner_prod(y);
		}, make_is(), *this, x);

		return future_transform{
			future_vector{std::move(futs)},
			[](auto && v) {
				return std::apply([](auto ...vs) {
					return (vs + ...);
				}, v);
			}};
	}

	std::tuple<Vecs...> vecs;

protected:
	template<std::size_t I, class F, class ... Multis>
	constexpr decltype(auto) apply_aux(F && f, Multis&& ... ms) const {
		return std::invoke(std::forward<F>(f),
		                   std::forward<Multis>(ms).template get<I>()...);
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

namespace std {

template <class... Vecs>
struct tuple_size<flecsi::linalg::multivector<Vecs...>> {
	static constexpr size_t value = sizeof...(Vecs);
};

template <std::size_t I, class... Vecs>
struct tuple_element<I, flecsi::linalg::multivector<Vecs...>> {
	using type = typename tuple_element<I, tuple<Vecs...>>::type;
};

}
