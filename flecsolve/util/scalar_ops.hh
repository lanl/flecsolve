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
#ifndef FLECSI_LINALG_UTIL_SCALAR_OPS_HH
#define FLECSI_LINALG_UTIL_SCALAR_OPS_HH

#include <flecsi/execution.hh>

namespace flecsolve::scalar_ops {

template<class T>
struct axpy {
	constexpr T operator()(T x) const noexcept { return a * x + y; }
	T a{0}, y{0};
};

template<class T>
struct identity {
	constexpr T operator()(T v) const noexcept { return v; }
};

template<class F, class G>
struct compose {
	template<class T>
	constexpr auto operator()(T v) const noexcept {
		return g(f(v));
	}
	F f;
	G g;
};
}

namespace flecsolve {
template<class T>
auto operator*(T v, flecsi::future<T> f) {
	return future_transform{std::move(f), scalar_ops::axpy<T>{v, 0}};
}

template<class T>
auto operator*(flecsi::future<T> f, T v) {
	return future_transform{std::move(f), scalar_ops::axpy<T>{v, 0}};
}

}

#endif
