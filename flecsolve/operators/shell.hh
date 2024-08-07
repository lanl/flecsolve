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
#ifndef FLECSOLVE_OPERATORS_SHELL_H
#define FLECSOLVE_OPERATORS_SHELL_H

#include <algorithm>

#include "flecsolve/operators/core.hh"
#include "flecsolve/operators/handle.hh"

namespace flecsolve::op {

template<class F, class ivar_t, class ovar_t>
struct shell : base<std::nullptr_t, ivar_t, ovar_t> {

	constexpr shell(F f, ivar_t, ovar_t) : f(std::move(f)) {}

	template<class domain_vec, class range_vec>
	constexpr decltype(auto) apply(const domain_vec & x, range_vec & y) const {
		return f(x, y);
	}

protected:
	F f;
};

template<class F, auto I, auto O>
auto make_shell(F && f, variable_t<I>, variable_t<O>) { 
	return core<shell<std::decay_t<F>, variable_t<I>, variable_t<O>>>(
		std::forward<F>(f), variable<I>, variable<O>);
}

template<class F>
auto make_shell(F && f) {
	return make_shell(std::forward<F>(f),
	                  variable<anon_var::anonymous>, variable<anon_var::anonymous>);
}

template<class F, auto I, auto O>
auto make_shared_shell(F && f, variable_t<I>, variable_t<O>) {
	return make_shared<shell<F,
	                         variable_t<I>,
	                         variable_t<O>>>(std::forward<F>(f),
	                                         variable<I>, variable<O>);
}

template<class F>
auto make_shared_shell(F && f) {
	return make_shared_shell(std::forward<F>(f),
	                         variable<anon_var::anonymous>,
	                         variable<anon_var::anonymous>);
}

template<auto ivar, auto ovar>
auto make_identity(variable_t<ivar>, variable_t<ovar>) {
	return make_shared_shell([](const auto & x, auto & y) { y.copy(x); },
	                         variable<ivar>, variable<ovar>);
}

static inline const auto I =
	make_shared_shell([](const auto & x, auto & y) { y.copy(x); });

}

#endif
