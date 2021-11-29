#pragma once

#include <algorithm>

namespace flecsi::linalg {

template<class F>
struct shell_operator {

	constexpr shell_operator(F f) : f(std::move(f)) {}

	template<class domain_vec, class range_vec>
	constexpr void apply(const domain_vec & x, range_vec & y) const {
		f(x, y);
	}

	template<class domain_vec, class range_vec>
	constexpr void residual(const domain_vec & b, const range_vec & x,
	                        range_vec & r) const {
		f(x, r);
		r.subtract(b, r);
	}

protected:
	F f;
};

template <class F> shell_operator(F) -> shell_operator<F>;

}
