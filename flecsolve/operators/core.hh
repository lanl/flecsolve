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
#ifndef FLECSOLVE_OPERATORS_CORE_HH
#define FLECSOLVE_OPERATORS_CORE_HH

#include <initializer_list>
#include <type_traits>

#include "traits.hh"
#include "flecsolve/util/traits.hh"
#include "flecsolve/vectors/variable.hh"
#include "flecsolve/vectors/traits.hh"

namespace flecsolve::op {

template<class Params = std::nullptr_t,
         class ivar = variable_t<anon_var::anonymous>,
         class ovar = variable_t<anon_var::anonymous>>
struct base {
	static constexpr auto input_var = ivar{};
	static constexpr auto output_var = ovar{};
	using params_t = Params;

	template<class Head,
	         class... Tail,
	         std::enable_if_t<!std::is_same_v<std::decay_t<Head>, base>, bool> =
	             true,
	         std::enable_if_t<
				 std::is_constructible_v<params_t, Head, Tail...> ||
					 is_aggregate_initializable_v<params_t, Head, Tail...>,
				 bool> = true>
	base(Head && h, Tail &&... t)
		: params{std::forward<Head>(h), std::forward<Tail>(t)...} {}

	template<
		class Head,
		class... Tail,
		std::enable_if_t<std::is_constructible_v<params_t,
	                                             std::initializer_list<Head>,
	                                             Tail...>,
	                     bool> = true>
	base(std::initializer_list<Head> head, Tail &&... tail)
		: params{head, std::forward<Tail>(tail)...} {}

	template<class T = params_t,
	         class = std::enable_if_t<std::is_null_pointer_v<T>>>
	base() : params{nullptr} {}

	params_t & get_params() { return params; }
	const params_t & get_params() const { return params; }

	template<op::label tag, class T>
	auto get_parameters(const T &) const {
		return nullptr;
	}

	template<class T>
	void reset(const T &) const {}

	auto & get_operator() { return *this; }
	const auto & get_operator() const { return *this; }

protected:
	params_t params;
};

template<class P>
struct core : P {
	static constexpr auto input_var = P::input_var;
	static constexpr auto output_var = P::output_var;
	using params_t = typename P::params_t;
	using policy_type = P;

	template<class Head,
	         class... Tail,
	         std::enable_if_t<
				 !std::is_same_v<std::decay_t<Head>, core<P>>,
				 bool> = true,
	         std::enable_if_t<std::is_constructible_v<P, Head, Tail...>,
	                          bool> = true>
	core(Head && h, Tail &&... t)
		: P{std::forward<Head>(h), std::forward<Tail>(t)...} {}

	template<
		class Head,
		class... Tail,
		std::enable_if_t<std::is_constructible_v<P,
	                                             std::initializer_list<Head>,
	                                             Tail...>,
	                     bool> = true>
	core(std::initializer_list<Head> h, Tail &&... t)
		: P{h, std::forward<Tail>(t)...} {}

	template<class D,
	         class R,
	         std::enable_if_t<is_vector_v<D>, bool> = true,
	         std::enable_if_t<is_vector_v<R>, bool> = true>
	decltype(auto) apply(const D & x, R & y) const {
		return P::apply(x, y);
	}

	template<class D,
	         class R,
	         std::enable_if_t<is_vector_v<D>, bool> = true,
	         std::enable_if_t<is_vector_v<R>, bool> = true>
	decltype(auto) operator()(const D & x, R & y) const {
		return P::apply(x, y);
	}

	template<class B,
	         class X,
	         class R,
	         std::enable_if_t<is_vector_v<B>, bool> = true,
	         std::enable_if_t<is_vector_v<X>, bool> = true,
	         std::enable_if_t<is_vector_v<R>, bool> = true>
	void residual(const B & b, const X & x, R & r) const {
		apply(x, r);

		decltype(auto) bs = b.subset(output_var);
		decltype(auto) rs = r.subset(output_var);

		rs.subtract(bs, rs);
	}
};

template<class P>
auto make(P && p) {
	return core<std::decay_t<P>>(std::forward<P>(p));
}
template<class P, class ... Args>
auto make_shared1(Args && ... args) {
	return std::make_shared<core<P>>(std::forward<Args>(args)...);
}

template<class T>
struct is_operator : std::false_type {};

template<class P>
struct is_operator<core<P>> : std::true_type {};

template<class T>
inline constexpr bool is_operator_v = is_operator<T>::value;

}
#endif
