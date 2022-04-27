#pragma once

#include <tuple>
#include <utility>

namespace flecsi {
namespace linalg {
namespace discrete_operators {

#include <cassert>
#include <functional>
#include <tuple>

template<class VT, class T>
struct OpExpr;

template<auto... vars, class... Ps>
struct OpExpr<multivariable_t<vars...>, std::tuple<Ps...>> {
	std::tuple<Ps...> ops;

	OpExpr(Ps... ps) : ops(std::make_tuple(ps...)) {}

	template<class U, class V>
	constexpr void apply(const U & u, V & v) const {
		std::apply([&](const auto &... a) { (a.apply(u, v), ...); }, ops);
	}

	template<class F, class U, class V>
	constexpr void residual(const F & f, const U & u, V & v) const {
		this->apply(u, v);
		v.subtract(f, v);
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

	static constexpr auto input_var = multivariable<vars...>;
	static constexpr auto output_var = multivariable<vars...>;
};

template<auto... vars, class... Ps>
inline constexpr auto op_expr(multivariable_t<vars...>, Ps... ps) {
	return OpExpr<multivariable_t<vars...>, std::tuple<Ps...>>(ps...);
}

} // namespace discrete_operators
} // namespace linalg
} // namespace flecsi
