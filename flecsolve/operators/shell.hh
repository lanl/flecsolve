#ifndef FLECSOLVE_OPERATORS_SHELL_H
#define FLECSOLVE_OPERATORS_SHELL_H

#include <algorithm>

#include "flecsolve/operators/base.hh"

namespace flecsolve::op {

template<class F>
struct shell : base<shell<F>> {

	constexpr shell(F f) : f(std::move(f)) {}

	template<class domain_vec, class range_vec>
	constexpr void apply(const domain_vec & x, range_vec & y) const {
		f(x, y);
	}

protected:
	F f;
};
template<class F>
shell(F) -> shell<F>;

static inline const shell I([](const auto & x, auto & y) { y.copy(x); });

}
#endif
