#pragma once

#include <tuple>
#include <functional>

namespace flecsi::linalg {

template <class... Futures>
struct future_vector
{
	decltype(auto) get() {
		return std::apply([](Futures & ... futs) {
			return std::make_tuple(futs.get()...);
		}, futures);
	}

	void wait() {
		wait_aux();
	}

	~future_vector() {
		wait_aux();
	}

	std::tuple<Futures...> futures;

protected:
	constexpr void wait_aux() {
		std::apply([](Futures & ... futs) {
			(futs.wait(), ...);
		}, futures);
	}
};
template <class... Futures>
future_vector(std::tuple<Futures...>)->future_vector<Futures...>; //automatic in C++20


template <class Future, class F>
struct future_transform
{
	decltype(auto) get() {
		return f(fut.get());
	}

	void wait() {
		fut.wait();
	}

	~future_transform() {
		fut.wait();
	}

	Future fut;
	F f;
};
template <class Future, class F>
future_transform(Future,F)->future_transform<Future,F>; // automatic in C++20
}
