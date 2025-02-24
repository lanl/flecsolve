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
#ifndef FLECSOLVE_VECTORS_SEQ_H
#define FLECSOLVE_VECTORS_SEQ_H

#include <fstream>

#include "flecsi/execution.hh"
#include "flecsi/util/array_ref.hh"

#include "core.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve::vec {

template<class Scalar, auto V = anon_var::anonymous>
struct seq_config {
	using scalar = Scalar;
	using real = typename num_traits<scalar>::real;
	static constexpr bool is_complex = num_traits<scalar>::is_complex;
	using len_t = std::size_t;
	static constexpr auto var = variable<V>;
	using var_t = decltype(V);
	static constexpr std::size_t num_components = 1;
};

template<class Config, template<class> class storage>
struct seq_data {
	using config = Config;
	using scalar = typename Config::scalar;
	using size_type = typename Config::len_t;

	seq_data() {}

	seq_data(storage<scalar> s) : store(std::move(s)) {}

	constexpr scalar & operator[](size_type i) { return store[i]; }

	constexpr const scalar & operator[](size_type i) const { return store[i]; }

	constexpr size_type size() const { return store.size(); }

	void resize(std::size_t s) { store.resize(s); }

	friend bool operator==(const seq_data & d1, const seq_data & d2) {
		return d1.store == d2.store;
	}
	friend bool operator!=(const seq_data & d1, const seq_data & d2) {
		return d1.store != d2.store;
	}

private:
	storage<scalar> store;
};

template<class T>
using vec_t = std::vector<T>;
template<class Config>
using seq_data_vec = seq_data<Config, vec_t>;
template<class T>
using span_t = flecsi::util::span<T>;
template<class Config>
using seq_data_span = seq_data<Config, span_t>;

template<class Data>
struct seq_ops {
	using config = typename Data::config;
	using scalar = typename config::scalar;
	using real = typename config::real;
	static constexpr bool is_complex = config::is_complex;
	using len_t = typename config::len_t;

	template<class T>
	struct fut {
		fut(T v) : val{v} {}
		T val;
		T get() { return val; }
	};

	template<class Src, class Dest>
	static void copy(const Src & src, Dest & dest) {
		fordofs([&](len_t i) { dest[i] = src[i]; }, src, dest);
	}

	template<class D>
	static void zero(D & x) {
		fordofs([&](len_t i) { x[i] = 0.; }, x);
	}

	template<class D>
	static void set_random(D & x, unsigned seed) {
		std::mt19937 gen(seed);
		std::uniform_real_distribution<real> dis(0., 1.);
		fordofs(
			[&](len_t i) {
				if constexpr (is_complex)
					x[i] = scalar(dis(gen), dis(gen));
				else
					x[i] = dis(gen);
			},
			x);
	}

	template<class D>
	static void set_to_scalar(scalar alpha, D & x) {
		fordofs([&](len_t i) { x[i] = alpha; }, x);
	}

	template<class D>
	static void scale(scalar alpha, D & x) {
		fordofs([&](len_t i) { x[i] *= alpha; }, x);
	}

	template<class D0, class D1>
	static void scale(scalar alpha, const D0 & x, D1 & y) {
		fordofs([&](len_t i) { y[i] = alpha * x[i]; }, x, y);
	}

	template<class X, class Y, class Z>
	static void add(const X & x, const Y & y, Z & z) {
		fordofs([&](len_t i) { z[i] = x[i] + y[i]; }, x, y, z);
	}

	template<class X, class Y, class Z>
	static void subtract(const X & x, const Y & y, Z & z) {
		fordofs([&](len_t i) { z[i] = x[i] - y[i]; }, x, y, z);
	}

	template<class X, class Y, class Z>
	static void multiply(const X & x, const Y & y, Z & z) {
		fordofs([&](len_t i) { z[i] = x[i] * y[i]; }, x, y, z);
	}

	template<class X, class Y, class Z>
	static void divide(const X & x, const Y & y, Z & z) {
		fordofs([&](len_t i) { z[i] = x[i] / y[i]; }, x, y, z);
	}

	template<class X, class Y>
	static void reciprocal(const X & x, Y & y) {
		fordofs([&](len_t i) { x[i] = 1. / y[i]; }, x, y);
	}

	template<class X>
	static void dump(std::string_view pre, const X & x) {
		std::string fname{pre};
		fname += "-" + std::to_string(flecsi::color());
		std::ofstream ofile{fname};
		fordofs([&](len_t i) { ofile << x[i] << '\n'; }, x);
	}

	template<class X, class Y, class Z>
	static void
	linear_sum(scalar alpha, const X & x, scalar beta, const Y & y, Z & z) {
		fordofs([&](len_t i) { z[i] = alpha * x[i] + beta * y[i]; }, x, y, z);
	}

	template<class X, class Y, class Z>
	static void axpy(scalar alpha, const X & x, const Y & y, Z & z) {
		fordofs([&](len_t i) { z[i] = alpha * x[i] + y[i]; }, x, y, z);
	}

	template<class X, class Z>
	static void axpby(scalar alpha, scalar beta, const X & x, Z & z) {
		fordofs([&](len_t i) { z[i] = alpha * x[i] + beta * z[i]; }, x, z);
	}

	template<class X, class Y>
	static void abs(const X & x, Y & y) {
		fordofs([&](len_t i) { y[i] = std::abs(x[i]); }, x, y);
	}

	template<class X, class Y>
	static void add_scalar(const X & x, scalar alpha, Y & y) {
		fordofs([&](len_t i) { y[i] = x[i] + alpha; }, x, y);
	}

	template<class X>
	static fut<real> min(const X & x) {
		auto curr = std::numeric_limits<real>::max();
		fordofs(
			[&](len_t i) {
				if constexpr (is_complex)
					curr = std::min(x[i].real(), curr);
				else
					curr = std::min(x[i], curr);
			},
			x);

		return curr;
	}

	template<class X>
	static fut<real> max(const X & x) {
		auto curr = std::numeric_limits<real>::lowest();
		fordofs(
			[&](len_t i) {
				if constexpr (is_complex)
					curr = std::max(x[i].real(), curr);
				else
					curr = std::max(x[i], curr);
			},
			x);

		return curr;
	}

	template<unsigned short p, class X>
	static fut<real> lp_norm(const X & x) {
		real curr = 0;
		if constexpr (p == 2) {
			if constexpr (is_complex)
				curr = scalar_prod(x, x).real();
			else
				curr = scalar_prod(x, x);
		}
		else {
			fordofs(
				[&](len_t i) {
					if constexpr (p == 1)
						curr += std::abs(x[i]);
					else
						curr += std::abs(x[i]);
				},
				x);
		}

		if constexpr (p == 1)
			return curr;
		else if constexpr (p == 2)
			return std::sqrt(curr);
		else
			return std::pow(curr, 1. / p);
	}

	template<class X>
	static fut<real> inf_norm(const X & x) {
		auto ret = std::numeric_limits<real>::min();
		fordofs([&](len_t i) { ret = std::max(std::abs(x[i]), ret); }, x);
		return ret;
	}

	template<class X, class Y>
	static fut<scalar> dot(const X & x, const Y & y) {
		return scalar_prod(x, y);
	}

	template<class X>
	static len_t local_size(const X & x) {
		return x.data.size();
	}

	template<class X>
	static fut<len_t> global_size(const X & x) {
		return local_size(x);
	}

	template<class X>
	static scalar & retreive(X & x, len_t i) {
		return x[i];
	}

	template<class X>
	static const scalar & retreive(const X & x, len_t i) {
		return x[i];
	}

protected:
	template<class X, class Y>
	static scalar scalar_prod(const X & x, const Y & y) {
		scalar res = 0.0;
		fordofs(
			[&](len_t i) {
				if constexpr (is_complex)
					res += std::conj(y[i]) * x[i];
				else
					res += x[i] * y[i];
			},
			x,
			y);
		return res;
	}

	template<class F, class A0, class... A>
	static void fordofs(F && f, const A0 & a0, const A &... arest) {
		len_t sz = a0.size();
		([&](const auto & a) { flog_assert(a.size() == sz, ""); }(arest), ...);
		for (len_t i = 0; i < sz; ++i) {
			f(i);
		}
	}
};

template<class T, auto V = anon_var::anonymous>
struct seq_vec : core<seq_data_vec, seq_ops, seq_config<T, V>> {
	using base = core<seq_data_vec, seq_ops, seq_config<T, V>>;
	seq_vec(variable_t<V>) : base{vec_t<T>{}} {}
	seq_vec() : base{vec_t<T>{}} {}
	seq_vec(variable_t<V>, std::size_t sz) : base{vec_t<T>(sz)} {}
	seq_vec(std::size_t sz) : base{vec_t<T>(sz)} {}
};

template<class T, auto V>
struct seq_view : core<seq_data_span, seq_ops, seq_config<T, V>> {
	using base = core<seq_data_span, seq_ops, seq_config<T, V>>;
	seq_view(span_t<T> span) : base{span} {}
	seq_view(variable_t<V>, span_t<T> span) : base{span} {}
};
template<class T>
seq_view(span_t<T>) -> seq_view<T, anon_var::anonymous>;
template<auto V, class T>
seq_view(variable_t<V>, span_t<T>) -> seq_view<T, V>;

template<class T, std::size_t nwork, auto var = anon_var::anonymous>
struct seq_work {
	using value_type = seq_vec<T, var>;
	seq_work(const seq_vec<T, var> & v) : csize{v.data.size()} {}

	template<std::size_t I>
	seq_vec<T, var> & get() {
		static_assert(I < nwork);
		if (vecs[I].data.size() != csize)
			vecs[I].data.resize(csize);

		return vecs[I];
	}

protected:
	std::size_t csize;
	std::array<seq_vec<T, var>, nwork> vecs;
};

}

namespace std {

template<class T, std::size_t nwork, auto var>
struct tuple_size<flecsolve::vec::seq_work<T, nwork, var>> {
	static constexpr size_t value = nwork;
};

template<std::size_t I, class T, std::size_t nwork, auto var>
struct tuple_element<I, flecsolve::vec::seq_work<T, nwork, var>> {
	using type = flecsolve::vec::seq_vec<T, var>;
};

}

#endif
