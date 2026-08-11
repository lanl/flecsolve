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
#pragma once

#include <flecsi/execution.hh>

#include <tuple>
#include <functional>
namespace flecsolve {

// for now, only consider scalar futures
template<class T>
using future = flecsi::future<T, flecsi::exec::launch_type_t::single>;

template<class... Futures>
struct future_vector {
	decltype(auto) get() {
		return std::apply(
			[](Futures &... futs) { return std::make_tuple(futs.get()...); },
			futures);
	}

	void wait() { wait_aux(); }

	~future_vector() { wait_aux(); }

	std::tuple<Futures...> futures;

protected:
	constexpr void wait_aux() {
		std::apply([](Futures &... futs) { (futs.wait(), ...); }, futures);
	}
};
template<class... Futures>
future_vector(std::tuple<Futures...>)
	-> future_vector<Futures...>; // automatic in C++20

template<class Future, class F>
struct future_transform {
	decltype(auto) get() { return f(fut.get()); }

	void wait() { fut.wait(); }

	~future_transform() { fut.wait(); }

	Future fut;
	F f;
};
template<class Future, class F>
future_transform(Future, F)
	-> future_transform<Future, F>; // automatic in C++20

template<class T, class Scalar>
struct is_scalar_or_future_impl : std::false_type {
};

template<class T, class Scalar>
struct is_scalar_or_future_impl<future<T>, Scalar>
	: std::bool_constant<std::is_convertible_v<T, Scalar>> {
};

template<class T, class F, class Scalar>
struct is_scalar_or_future_impl<future_transform<future<T>, F>, Scalar>
	: std::bool_constant<std::is_convertible_v<T, Scalar>> {
};

template<class T, class Scalar>
inline constexpr bool is_scalar_or_future_v =
	std::is_convertible_v<T, Scalar> ||
	is_scalar_or_future_impl<T, Scalar>::value;

}
