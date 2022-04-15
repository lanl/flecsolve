#pragma once

#include <tuple>
#include <utility>

namespace flecsi {
namespace linalg {
namespace discrete_operators {

#include <cassert>
#include <functional>
#include <tuple>

template<class T>
struct OpExpr;

template<class... Ps>
struct OpExpr<std::tuple<Ps...>> {
	std::tuple<Ps...> ops;

	OpExpr(Ps... ps) : ops(std::make_tuple(ps...)) {}

	template<class U, class V>
	constexpr void apply(U && u, V && v) const {
		std::apply(
			[&](auto &&... a) {
				(a.apply(std::forward<decltype(u)>(u),
			             std::forward<decltype(v)>(v)),
			     ...);
			},
			ops);
	}

	template<class F, class U, class V>
	constexpr void residual(F && f, U && u, V && v) const {
		this->apply(std::forward<decltype(u)>(u), std::forward<decltype(v)>(v));
		v.subtract(std::forward<decltype(f)>(f), std::forward<decltype(v)>(v));
	}

	constexpr decltype(auto) get_parameters() const {
		return _gp(std::make_index_sequence<sizeof...(Ps)>{});
	}

	template<std::size_t... II>
	constexpr decltype(auto) _gp(std::index_sequence<II...>) const {
		return std::make_tuple(
			typename std::decay_t<
				std::tuple_element_t<II, decltype(ops)>>::param_type{}...);
	}
	// reset
};

template<class... Ps>
inline constexpr auto op_expr(Ps... ps) {
	return OpExpr<std::tuple<Ps...>>(ps...);
}

} // namespace discrete_operators
} // namespace linalg
} // namespace flecsi