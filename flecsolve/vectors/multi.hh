#ifndef FLECSI_LINALG_VECTORS_MULTI_H
#define FLECSI_LINALG_VECTORS_MULTI_H

#include <tuple>
#include <type_traits>

#include "base.hh"
#include "flecsolve/vectors/variable.hh"
#include "operations/multi.hh"

namespace flecsolve::vec {

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

template<class... Vecs>
using multivector_scalar = typename std::tuple_element<
	0,
	std::tuple<std::remove_reference_t<Vecs>...>>::type::scalar;
template<class... Vecs>
using multivector_len = typename std::
	tuple_element<0, std::tuple<std::remove_reference_t<Vecs>...>>::type::len_t;
template<class... Vecs>
using multivector_data = std::tuple<Vecs...>;
template<class... Vecs>
using multivector_ops = ops::multi<multivector_scalar<Vecs...>,
                                   multivector_len<Vecs...>,
                                   multivector_data<Vecs...>,
                                   sizeof...(Vecs)>;

template<class VarType, class... Vecs>
struct multi : base<multi<VarType, Vecs...>> {
	using base_t = base<multi<VarType, Vecs...>>;
	using base_t::data;
	static constexpr std::size_t num_components = sizeof...(Vecs);

	template<
		class... VT,
		typename = std::enable_if_t<
			(... && std::is_same_v<typename std::remove_reference_t<VT>::var_t,
	                               VarType>)>>
	multi(VT &&... vs)
		: base_t{multivector_data<Vecs...>{std::forward<VT>(vs)...}} {}

	template<std::size_t I>
	constexpr auto & get() & {
		return std::get<I>(data);
	}

	template<std::size_t I>
	constexpr const auto & get() const & {
		return std::get<I>(data);
	}

	template<VarType var, std::size_t I>
	constexpr decltype(auto) get() const {
		using tuple_t = std::tuple<std::remove_reference_t<Vecs>...>;
		using curr = typename std::tuple_element<I, tuple_t>::type;
		if constexpr (curr::var == variable<var>)
			return std::get<I>(data);
		else if constexpr (I + 1 == sizeof...(Vecs)) {
			static_assert(I + 1 < sizeof...(Vecs));
			return nullptr;
		}
		else
			return get<var, I + 1>();
	}

	template<VarType var, std::size_t I>
	constexpr decltype(auto) get() {
		return as_mutable(std::as_const(*this).template get<var, I>());
	}

	template<VarType var>
	constexpr decltype(auto) getvar() const {
		return get<var, 0>();
	}

	template<VarType var>
	constexpr decltype(auto) getvar() {
		return get<var, 0>();
	}

	template<VarType var>
	constexpr decltype(auto) subset_impl(variable_t<var>) const {
		return getvar<var>();
	}

	template<VarType var>
	constexpr decltype(auto) subset_impl(variable_t<var>) {
		return getvar<var>();
	}

	template<VarType... vars>
	constexpr decltype(auto) subset_impl(multivariable_t<vars...>) const {
		if constexpr (sizeof...(vars) == 1) {
			return getvar<vars...>();
		}
		else {
			return multi<VarType, decltype(getvar<vars>())...>(
				getvar<vars>()...);
		}
	}

	template<VarType... vars>
	constexpr decltype(auto) subset_impl(multivariable_t<vars...>) {
		if constexpr (sizeof...(vars) == 1) {
			return getvar<vars...>();
		}
		else {
			return multi<VarType, decltype(getvar<vars>())...>(
				getvar<vars>()...);
		}
	}

	template<class F>
	constexpr decltype(auto) apply_impl(F && f) {
		return std::apply(std::forward<F>(f), data);
	}

	static constexpr auto var =
		multivariable<std::remove_reference_t<Vecs>::var.value...>;
};

template<class... VT>
multi(VT &&... vs) -> multi<
	typename std::tuple_element<0, std::tuple<std::remove_reference_t<VT>...>>::
		type::var_t,
	VT...>;

template<class VarType, class... V0, class... V1>
bool operator==(const multi<VarType, V0...> & v0,
                const multi<VarType, V1...> & v1) {
	return v0.data == v1.data;
}
template<class VarType, class... V0, class... V1>
bool operator!=(const multi<VarType, V0...> & v0,
                const multi<VarType, V1...> & v1) {
	return v0.data != v1.data;
}

}

namespace flecsolve {
template<class VarType, class... Vecs>
struct traits<vec::multi<VarType, Vecs...>> {
	// static constexpr auto var =
	// multivariable<std::remove_reference_t<Vecs>::var.value...>;
	static constexpr auto var = variable<anon_var::anonymous>;
	using data_t = vec::multivector_data<Vecs...>;
	using ops_t = vec::multivector_ops<Vecs...>;
};
}

namespace std {

template<class VarType, class... Vecs>
struct tuple_size<flecsolve::vec::multi<VarType, Vecs...>> {
	static constexpr size_t value = sizeof...(Vecs);
};

template<std::size_t I, class VarType, class... Vecs>
struct tuple_element<I, flecsolve::vec::multi<VarType, Vecs...>> {
	using type = typename tuple_element<I, tuple<Vecs...>>::type;
};

}
#endif
