#pragma once

#include <tuple>
#include <bits/utility.h>
#include <utility>
#include <cassert>
#include <functional>
#include <tuple>

#include "flecsolve/physics/common/operator_utils.hh"
#include "flecsi/util/constant.hh"

namespace flecsolve {
namespace physics {

template<auto... V>
using has = flecsi::util::constants<V...>;

template<class T>
inline constexpr auto start(T && t) {
	return std::forward<T>(t);
}

template<class VT, class T, class K = utils::mp::iota<std::tuple_size_v<T>>>
struct OpExpr;

template<auto... vars, class... Ps, int... Is>
struct OpExpr<multivariable_t<vars...>, std::tuple<Ps...>, has<Is...>> {
	std::tuple<Ps...> ops;

	OpExpr(Ps... ps) : ops(std::make_tuple(ps...)) {}

	template<class _T>
	struct Flat {
		_T ops;
		constexpr decltype(auto) operator*() {
			return (start(*std::get<Is>(ops)), ...);
		}
	};

	template<class... P_>
	inline constexpr static auto flat(P_ &&... p) {
		return Flat<std::tuple<P_...>>{
			std::tuple<P_...>{std::forward<P_>(p)...}};
	}

	// set/get
	template<class J>
	constexpr decltype(auto) at(J const & i) {
		return (start(std::get<Is>(this->ops).at(i)), ...);
	}

	template<class J>
	constexpr decltype(auto) at(J const & i) const {
		return (start(std::get<Is>(this->ops).at(i)), ...);
	}

	// returns flattened expression (all nested tuples expanded)
	constexpr decltype(auto) flat() {
		return flat(std::get<Is>(this->ops).flat()...);
	}

	operator decltype (*(flat(std::get<Is>(ops).flat()...)))() {
		return *flat();
	}

	template<class U, class V>
	constexpr void apply(const U & u, V & v) const {
		std::apply([&](const auto &... a) { (a.apply(u, v), ...); }, ops);
	}

	template<class F, class U, class V>
	constexpr void residual(const F & f, const U & u, V & v) const {
		this->apply(u, v);
		v.subtract(f, v);
	}

	template<size_t I = 0, class... Pars>
	constexpr void reset(std::tuple<Pars...> op_pars) {
		this->at(I) = std::get<I>(op_pars);
		if constexpr (I + 1 != sizeof...(Pars)) {
			reset<I + 1>(op_pars);
		}
	}

	template<auto CPH, class VPH>
	constexpr decltype(auto) get_parameters(VPH & v) const {
		return std::make_tuple(this->at(Is)...);
	}

	static constexpr auto input_var = multivariable<vars...>;
	static constexpr auto output_var = multivariable<vars...>;
};

template<auto... vars, class... Ps>
inline constexpr auto op_expr(multivariable_t<vars...>, Ps... ps) {
	return OpExpr<multivariable_t<vars...>, std::tuple<Ps...>>(ps...);
}


template<class... Ps>
inline constexpr auto op_expr(Ps... ps) {
	return OpExpr<multivariable_t<>, std::tuple<Ps...>>(ps...);
}

}
}
