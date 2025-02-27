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

namespace flecsolve::op {

template<class F>
struct shell : base<> {

	constexpr shell(F f) : f(std::move(f)) {}

	template<class domain_vec, class range_vec>
	constexpr decltype(auto) apply(const domain_vec & x, range_vec & y) const {
		return f(x, y);
	}

protected:
	F f;
};
template<class F>
shell(F) -> shell<F>;

template<class F>
auto make_shell(F && f) {
	return core<shell<F>>(std::forward<F>(f));
}

static inline const auto I =
	make_shell([](const auto & x, auto & y) { y.copy(x); });

}
#endif
