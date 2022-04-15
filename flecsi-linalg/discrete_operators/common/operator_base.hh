#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <flecsi/util/array_ref.hh>
#include <flecsi/util/constant.hh>
#include <tuple>
#include <type_traits>
#include <utility>

namespace flecsi {
namespace linalg {
namespace discrete_operators {

template<class Derived>
struct operator_traits;
template<class Derived>
struct operator_parameters;

template<class Derived, template<class> class BaseType>
struct operator_base {
	using exact_type = Derived;

	constexpr exact_type const & derived() const {
		return static_cast<exact_type const &>(*this);
	}

	constexpr exact_type & derived() {
		return static_cast<exact_type &>(*this);
	}

	template<class U, class V>
	constexpr decltype(auto) apply(U && u, V && v) const {
		return derived().apply(std::forward<U>(u), std::forward<V>(v));
	}

private:
	operator_base() {}
	friend BaseType<Derived>;
};

template<class Derived>
struct operator_settings : operator_base<Derived, operator_settings> {

	using base_type = operator_base<Derived, operator_settings>;
	using exact_type = typename base_type::exact_type;
	using param_type = operator_parameters<Derived>;

	template<class P>
	operator_settings(P && p) : base_type() {
		this->reset(p);
	}

	template<class... Ps>
	constexpr static param_type getParamType(Ps &&... ps) {
		return param_type{ps...};
	}

	template<class P>
	constexpr decltype(auto) reset(P && pars) {
		parameters = pars;
	}

private:
	param_type parameters;
	friend exact_type;
};

template<template<class> class... Mixins>
struct make_mixins {
	template<class Derived>
	struct templ : Mixins<Derived>... {
		template<class... Args>
		void exec(Args &&... args) {
			(Mixins<Derived>::exec(std::forward<Args>(args)...), ...);
		}
	};
};

template<class ML>
struct execute_mixins;

template<class Derived, template<class> class MixinSet>
struct execute_mixins<MixinSet<Derived>>
	: operator_base<Derived, execute_mixins>, MixinSet<Derived> {
	using base_type = operator_base<Derived, execute_mixins>;
	using exact_type = typename base_type::exact_type;
	using mixset_type = MixinSet<Derived>;

	template<class... Args>
	void exec(Args &&... args) {
		mixset_type::exec(std::forward<Args>(args)...);
	}
};

// template <class TopHost, class Mixes, class Tuple, std::size_t... ii>
// constexpr decltype(auto) make_ophost_impl(std::index_sequence<ii...>,
//                                           Tuple &&tuple) {
//   return TopHost(std::make_from_tuple<std::tuple_element_t<ii, Mixes>>(
//       std::get<ii>(std::forward<Tuple>(tuple)))...);
// }

// template <template <template <class> class...> class HT,
//           template <class> class... Mixes, class... Tuples>
// constexpr decltype(auto) make_ophost(Tuples &&...tuples) {
//   static_assert(sizeof...(Mixes) == sizeof...(Tuples));
//   using TopHost = HT<Mixes...>;
//   return make_ophost_impl<TopHost, std::tuple<Mixes<TopHost>...>>(
//       std::make_index_sequence<sizeof...(Mixes)>{},
//       std::make_tuple(std::forward<Tuples>(tuples)...));
// }

template<class Op, class... Ps>
inline auto make_operator(Ps &&... ps) {
	auto pars = Op::getParamType(std::forward<Ps>(ps)...);
	return Op(pars);
}

template<typename>
struct is_tuple : std::false_type {};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

} // namespace discrete_operators
} // namespace linalg
} // namespace flecsi
