#pragma once

#include <algorithm>

#include <flecsi-linalg/vectors/variable.hh>

namespace flecsi::linalg::op {

template<class F,
         auto ivar = anon_var::anonymous,
         auto ovar = anon_var::anonymous>
struct shell {

	constexpr shell(F f) : f(std::move(f)) {}

	template<class domain_vec, class range_vec>
	constexpr void apply(const domain_vec & x, range_vec & y) const {
		f(x, y);
	}

	template<class domain_vec, class range_vec>
	constexpr void
	residual(const domain_vec & b, const range_vec & x, range_vec & r) const {
		f(x, r);
		r.subtract(b, r);
	}

	static constexpr auto input_var = variable<ivar>;
	static constexpr auto output_var = variable<ovar>;

protected:
	F f;
};
template<class F>
shell(F) -> shell<F>;

static inline const shell I([](const auto & x, auto & y) { y.copy(x); });

}
