#pragma once

#include <tuple>

#include "vector.hh"
#include "operations/multi.hh"


namespace flecsi::linalg::vec {

template <class... Vecs> using multivector_scalar =
	typename std::tuple_element<0, std::tuple<std::remove_reference_t<Vecs>...>>::type::scalar;
template <class... Vecs> using multivector_data = std::tuple<Vecs...>;
template <class... Vecs>
using multivector_ops =
	ops::multi<vector_types<multivector_scalar<Vecs...>>,
	           multivector_data<Vecs...>, sizeof...(Vecs)>;

template <class... Vecs>
using multivector_base =
	vector<multivector_data<Vecs...>,
	       multivector_ops<Vecs...>>;

template <class VarType, class... Vecs>
struct multi : public multivector_base<Vecs...>
{
	using base = multivector_base<Vecs...>;
	using base::data;

	template<class ... VT,
	         typename = std::enable_if_t<
	         (... && std::is_same_v<typename std::remove_reference_t<VT>::var_t, VarType>)>>
	multi(VT&&... vs) :
		base{multivector_data<Vecs...>{std::forward<VT>(vs)...}} {}

	template<std::size_t I>
	constexpr auto & get() & {
		return std::get<I>(data);
	}

	template<std::size_t I>
	constexpr const auto & get() const & {
		return std::get<I>(data);
	}

	template<VarType var, std::size_t I>
	constexpr decltype(auto) get() {
		using tuple_t = std::tuple<std::remove_reference_t<Vecs>...>;
		using curr = typename std::tuple_element<I, tuple_t>::type;
		if constexpr (curr::var == var)
			return std::get<I>(data);
		else if constexpr (I + 1 == sizeof...(Vecs))  {
			static_assert(I+1 < sizeof...(Vecs));
			return nullptr;
		} else
			return get<var, I+1>();
	}

	template<VarType var>
	constexpr decltype(auto) getvar() {
		return get<var, 0>();
	}

	template<VarType ... vars>
	constexpr auto subset() {
		return multi<VarType,
			decltype(getvar<vars>())...>(getvar<vars>()...);
	}
};

template <class... VT>
multi(VT&&... vs)->multi<typename std::tuple_element<0, std::tuple<std::remove_reference_t<VT>...>>::type::var_t, VT...>;

}

namespace std {

template <class VarType, class... Vecs>
struct tuple_size<flecsi::linalg::vec::multi<VarType, Vecs...>> {
	static constexpr size_t value = sizeof...(Vecs);
};

template <std::size_t I, class VarType, class... Vecs>
struct tuple_element<I, flecsi::linalg::vec::multi<VarType, Vecs...>> {
	using type = typename tuple_element<I, tuple<Vecs...>>::type;
};

}
