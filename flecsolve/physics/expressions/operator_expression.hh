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


	template<class U, class V>
	constexpr void apply(const U & u, V & v) const {
		std::apply([&](const auto &... a) { (a.apply(u, v), ...); }, ops);
	}

	template<class F, class U, class V>
	constexpr void residual(const F & f, const U & u, V & v) const {
		this->apply(u, v);
		v.subtract(f, v);
	}

	constexpr decltype(auto) flat() const{
		return std::tuple_cat(std::apply([&](auto...a){
		  return std::tuple_cat(a.flat()...);}, ops));
	  }


	template<class... Pars>
	constexpr void reset(std::tuple<Pars...> op_pars) {
		_res(op_pars, flat());
	}

	template<std::size_t I = 0, class ...Pars, class T>
	constexpr void _res(std::tuple<Pars...> opp, T t)
	{
		std::get<I>(t).reset(std::get<I>(opp));
		if constexpr (I + 1 != sizeof...(Pars))
		{
			_res<I+1>(opp, t);
		}
	}

	template<auto CPH, class VPH>
	constexpr decltype(auto) get_parameters(VPH & v) const {
		//return std::make_tuple(this->at(Is)...);
		return std::apply([&](const auto&...a){ return std::make_tuple(a.template get_parameters<CPH>(v)...);}, flat());
	}

	const std::string to_string() const
	{
		std::ostringstream ss;
		ss << "[";
		//std::apply([&](const auto& ...a){ (ss << ... << a.to_string());}, flat());
		std::apply([&](const auto& ...a){ ((ss << a.to_string() << " "), ...);}, ops);
		ss << "]";
		return ss.str();
	}

	static constexpr auto input_var = multivariable<vars...>;
	static constexpr auto output_var = multivariable<vars...>;
};

template<auto... vars, class... Ps>
inline constexpr auto op_expr(multivariable_t<vars...>, Ps... ps) {
	return OpExpr<multivariable_t<vars...>, std::tuple<Ps...>>(ps...);
}



}
}
