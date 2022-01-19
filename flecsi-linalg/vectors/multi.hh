#pragma once

#include <tuple>

#include "vector.hh"
#include "operations/multi.hh"


namespace flecsi::linalg::vec {

template <class... Vecs> using multivector_scalar =
	typename std::tuple_element<0, std::tuple<Vecs...>>::type::scalar;
template <class... Vecs> using multivector_data = std::tuple<Vecs...>;
template <class... Vecs>
using multivector_ops =
	ops::multi<vector_types<multivector_scalar<Vecs...>>, Vecs...>;

template <class... Vecs>
using multivector_base =
	vector<multivector_data<Vecs...>,
	       multivector_ops<Vecs...>>;

template <class... Vecs>
struct multi : public multivector_base<Vecs...>
{
	using base = multivector_base<Vecs...>;
	using base::data;

	multi(Vecs... vs) :
		base{{std::move(vs)...}} {}

	template<std::size_t I>
	constexpr auto & get() & {
		return std::get<I>(data);
	}

	template<std::size_t I>
	constexpr const auto & get() const & {
		return std::get<I>(data);
	}

};

}

namespace std {

template <class... Vecs>
struct tuple_size<flecsi::linalg::vec::multi<Vecs...>> {
	static constexpr size_t value = sizeof...(Vecs);
};

template <std::size_t I, class... Vecs>
struct tuple_element<I, flecsi::linalg::vec::multi<Vecs...>> {
	using type = typename tuple_element<I, tuple<Vecs...>>::type;
};

}
