#pragma once

#include <tuple>

#include "vector.hh"
#include "operations/multivector_operations.hh"


namespace flecsi::linalg {

template <class... Vecs> using multivector_real =
	typename std::tuple_element<0, std::tuple<Vecs...>>::type::real_t;
template <class... Vecs> using multivector_data = std::tuple<Vecs...>;
template <class... Vecs>
using multivector_ops =
	multivector_operations<multivector_real<Vecs...>, Vecs...>;

template <class... Vecs>
using multivector_base =
	vector<multivector_data<Vecs...>,
	       multivector_ops<Vecs...>,
	       multivector_real<Vecs...>>;

template <class... Vecs>
struct multivector : public multivector_base<Vecs...>
{
	using base = multivector_base<Vecs...>;
	using base::data;

	multivector(Vecs... vs) :
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
struct tuple_size<flecsi::linalg::multivector<Vecs...>> {
	static constexpr size_t value = sizeof...(Vecs);
};

template <std::size_t I, class... Vecs>
struct tuple_element<I, flecsi::linalg::multivector<Vecs...>> {
	using type = typename tuple_element<I, tuple<Vecs...>>::type;
};

}
