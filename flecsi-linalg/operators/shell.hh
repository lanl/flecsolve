#pragma once

#include <algorithm>

namespace flecsi::linalg::op {

template<class F>
struct shell {

	constexpr shell(F f) : f(std::move(f)) {}

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
template <class F> shell(F) -> shell<F>;

static inline const shell I([](const auto &x, auto &y) { y.copy(x); });

}
