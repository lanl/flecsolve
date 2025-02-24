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
#ifndef FLECSI_LINALG_VECTORS_MULTI_H
#define FLECSI_LINALG_VECTORS_MULTI_H

#include <tuple>
#include <type_traits>

#include "core.hh"
#include "traits.hh"
#include "flecsolve/vectors/variable.hh"
#include "operations/multi.hh"

namespace flecsolve::vec {

template<class... Vecs>
struct multi_config {
	// multivector scalar type is scalar type of first component vector
	using scalar = typename std::tuple_element_t<
		0,
		std::tuple<std::remove_reference_t<Vecs>...>>::scalar;
	using len_t = typename std::
		tuple_element_t<0, std::tuple<std::remove_reference_t<Vecs>...>>::len_t;
	using real = typename num_traits<scalar>::real;
	static constexpr auto var =
		multivariable<std::remove_reference_t<Vecs>::var.value...>;
	using var_t = typename std::
		tuple_element_t<0, std::tuple<std::remove_reference_t<Vecs>...>>::var_t;
	using storage_type = std::tuple<Vecs...>;
	static constexpr std::size_t num_components = sizeof...(Vecs);
};

namespace data {
template<class Config>
struct multi {
	using config = Config;
	template<class... V>
	multi(V &&... v) : components(std::forward<V>(v)...) {}
	typename Config::storage_type components;
};
}

namespace detail {
// https://stackoverflow.com/a/47369227
template<typename T>
constexpr T & as_mutable(T const & value) noexcept {
	return const_cast<T &>(value);
}
template<typename T>
constexpr T * as_mutable(T const * value) noexcept {
	return const_cast<T *>(value);
}
template<typename T>
constexpr T * as_mutable(T * value) noexcept {
	return value;
}
template<typename T>
void as_mutable(T const &&) = delete;
}

template<class... Vecs>
struct multi : core<data::multi, ops::multi, multi_config<Vecs...>> {
	using base = core<data::multi, ops::multi, multi_config<Vecs...>>;
	using base::data;
	using base::var;
	using var_t = typename base::var_t;
	using data_t = typename base::data_t;

	template<class Head,
	         class... Tail,
	         std::enable_if_t<
				 (... && std::is_same_v<
							 typename std::remove_reference_t<Head>::var_t,
							 typename std::remove_reference_t<Tail>::var_t>),
				 bool> = true>
	multi(Head && head, Tail &&... tail)
		: base{data_t{std::forward<Head>(head), std::forward<Tail>(tail)...}} {}

	template<std::size_t I>
	constexpr auto & get() & {
		return std::get<I>(data.components);
	}

	template<std::size_t I>
	constexpr const auto & get() const & {
		return std::get<I>(data.components);
	}

	template<var_t var, std::size_t I>
	constexpr decltype(auto) get() const {
		using tuple_t = std::tuple<std::remove_reference_t<Vecs>...>;
		using curr = typename std::tuple_element<I, tuple_t>::type;
		if constexpr (curr::var == variable<var>)
			return std::get<I>(data.components);
		else if constexpr (I + 1 == sizeof...(Vecs)) {
			static_assert(I + 1 < sizeof...(Vecs));
			return nullptr;
		}
		else
			return get<var, I + 1>();
	}

	template<var_t var, std::size_t I>
	constexpr decltype(auto) get() {
		return detail::as_mutable(std::as_const(*this).template get<var, I>());
	}

	template<var_t var>
	constexpr decltype(auto) getvar() const {
		return get<var, 0>();
	}

	template<var_t var>
	constexpr decltype(auto) getvar() {
		return get<var, 0>();
	}

	template<var_t var>
	constexpr decltype(auto) subset(variable_t<var>) const {
		return getvar<var>();
	}

	template<var_t var>
	constexpr decltype(auto) subset(variable_t<var>) {
		return getvar<var>();
	}

	template<var_t... vars>
	constexpr decltype(auto) subset(multivariable_t<vars...>) const {
		if constexpr (sizeof...(vars) == 1) {
			return getvar<vars...>();
		}
		else {
			return multi<decltype(getvar<vars>())...>(getvar<vars>()...);
		}
	}

	template<var_t... vars>
	constexpr decltype(auto) subset(multivariable_t<vars...>) {
		if constexpr (sizeof...(vars) == 1) {
			return getvar<vars...>();
		}
		else {
			return multi<decltype(getvar<vars>())...>(getvar<vars>()...);
		}
	}
};
template<class H, class... T>
multi(H &&, T &&... vs) -> multi<H, T...>;

template<class... V0, class... V1>
bool operator==(const multi<V0...> & v0, const multi<V1...> & v1) {
	return v0.data.components == v1.data.components;
}
template<class... V0, class... V1>
bool operator!=(const multi<V0...> & v0, const multi<V1...> & v1) {
	return v0.data.components != v1.data.components;
}

template<class... Vecs,
         std::enable_if_t<(... && is_vector_v<Vecs>), bool> = true>
auto make(Vecs &&... vecs) {
	return multi(std::forward<Vecs>(vecs)...);
}
}

namespace std {

template<class... Vecs>
struct tuple_size<flecsolve::vec::multi<Vecs...>> {
	static constexpr size_t value = sizeof...(Vecs);
};

template<std::size_t I, class... Vecs>
struct tuple_element<I, flecsolve::vec::multi<Vecs...>> {
	using type = typename tuple_element<I, tuple<Vecs...>>::type;
};

}
#endif
