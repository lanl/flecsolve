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
#pragma once

#include <tuple>
#include <utility>
#include <cassert>
#include <functional>
#include <tuple>

#include "flecsolve/physics/common/operator_utils.hh"
#include "flecsi/util/constant.hh"
#include "flecsolve/vectors/variable.hh"
#include "flecsolve/operators/core.hh"

namespace flecsolve {
namespace physics {

/// this is defined in flecsi/topo/core.hh, but is useful here.
/// alternatively, a custom `int_list<...>` would do the same role
template<auto... V>
using has = flecsi::util::constants<V...>;

/// helper for flat()
template<class T>
inline constexpr auto start(T && t) {
	return std::forward<T>(t);
}

/// forward declare
template<class VT, class T, class K = utils::mp::iota<std::tuple_size_v<T>>>
struct OpExpr;

/**
 * @brief represents a 'sentence' of operators
 *
 * `OpExpr` is an expression tree, with a simple syntax.
 * All nodes have the `operator` interface:
 * ```
 *   operator::apply(const Domain& in, Range& out) -> void
 *   operator::get_parameters() -> std::tuple<Parameters...>
 *   operator::reset(const Domain& in, Range& out) -> void
 * ```
 * see `common/operator_base.hh`
 *
 * Expressions allow for composable algorithms, and in future work
 * for a more powerful domain-specific language.
 */
template<auto... vars, class... Ps, int... Is>
struct OpExpr<multivariable_t<vars...>, std::tuple<Ps...>, has<Is...>>
	: op::base<std::tuple<Ps...>,
               multivariable_t<vars...>,
               multivariable_t<vars...>> {

	using base_t = op::base<std::tuple<Ps...>,
	                        multivariable_t<vars...>,
	                        multivariable_t<vars...>>;
	using base_t::params;

	OpExpr(Ps... ps) : base_t(std::make_tuple(ps...)) {}

	/// these are place-holder `apply()`s, and in the tree are the nodes
	template<class U, class V>
	constexpr void apply(const U & u, V & v) const {
		std::apply([&](const auto &... a) { (a.apply(u, v), ...); }, params);
	}

	template<class F, class U, class V>
	constexpr void residual_impl(const F & f, const U & u, V & v) const {
		this->apply(u, v);
		v.subtract(f, v);
	}

	/**
	 * @brief flatten/concretize the expression
	 *
	 * `OpExpr` is constructed as a tree, with placeholder nodes and concrete
	 * operators, but it is sometimes necessary to interacte with the operators
	 * only, such as retreiving operator parameters. This routine is meant
	 * to allow that.
	 */
	// TODO: should be protected, the interface shouldn't expose this ambiguity
	// TODO: this feels a little brutish, maybe isn't efficient either
	constexpr decltype(auto) flat() const {
		return std::tuple_cat(std::apply(
			[&](auto... a) { return std::tuple_cat(a.flat()...); }, params));
	}

	template<class... Pars>
	constexpr void reset(std::tuple<Pars...> op_pars) {
		_res(op_pars, flat());
	}

	/// helper for `residual()`
	// TODO: should this be protected?
	template<std::size_t I = 0, class... Pars, class T>
	constexpr void _res(std::tuple<Pars...> opp, T t) {
		std::get<I>(t).reset(std::get<I>(opp));
		if constexpr (I + 1 != sizeof...(Pars)) {
			_res<I + 1>(opp, t);
		}
	}

	template<auto CPH, class VPH>
	constexpr decltype(auto) get_parameters_impl(VPH & v) const {
		return std::apply(
			[&](const auto &... a) {
				return std::make_tuple(a.template get_parameters<CPH>(v)...);
			},
			flat());
	}

	const std::string to_string() const {
		std::ostringstream ss;
		ss << "[";
		std::apply(
			[&](const auto &... a) { ((ss << a.to_string() << " "), ...); },
			params);
		ss << "]";
		return ss.str();
	}
};

template<auto... vars, class... Ps>
inline constexpr auto op_expr(multivariable_t<vars...>, Ps... ps) {
	return op::core<OpExpr<multivariable_t<vars...>, std::tuple<Ps...>>>(ps...);
}

}
}
