#pragma once

#include <algorithm>

namespace flecsi::linalg {

template<class domain_vec, class range_vec, class F>
struct shell_operator {

	constexpr shell_operator(F && f) : f(std::forward<F>(f)) {}

	constexpr void apply(const domain_vec & x, range_vec & y) {
		f(x, y);
	}

	constexpr void residual(const domain_vec & b, const range_vec & x,
	                        range_vec & r) {
		f(x, r);
		r.subtract(b, r);
	}

protected:
	F f;
};

}
