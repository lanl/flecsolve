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
