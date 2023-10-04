#ifndef FLECSOLVE_VECTORS_SEQ_H
#define FLECSOLVE_VECTORS_SEQ_H

#include <fstream>

#include "flecsi/execution.hh"
#include "flecsi/util/array_ref.hh"

#include "base.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve::vec {

template<class Derived>
struct seq : base<Derived> {
	using base_t = base<Derived>;
	using scalar = typename base_t::scalar;
	using len_t = typename base_t::len_t;
	using base_t::data;

	seq() {}

	template<class D>
	FLECSI_INLINE_TARGET seq(D && d) : base_t{std::forward<D>(d)} {}

	constexpr scalar & operator[](len_t i) { return data[i]; }

	constexpr const scalar & operator[](len_t i) const { return data[i]; }
};

template<class T>
struct span_view {
	using span_t = flecsi::util::span<T>;
	using scalar = typename span_t::value_type;
	using size_type = typename span_t::size_type;

	span_view(span_t s) : span(s) {}

	constexpr scalar & operator[](size_type i) { return span[i]; }

	constexpr const scalar & operator[](size_type i) const { return span[i]; }

	constexpr size_type size() const { return span.size(); }

	span_t span;
};

template<class Scalar>
struct seq_ops {
	using scalar = Scalar;
	using real = typename num_traits<Scalar>::real;
	static constexpr bool is_complex = num_traits<scalar>::is_complex;
	using len_t = std::size_t;

	template<class T>
	struct fut {
		fut(T v) : val{v} {}
		T val;
		T get() { return val; }
	};

	template<class Src, class Dest>
	void copy(const Src & src, Dest & dest) const {
		fordofs([&](len_t i) { dest[i] = src[i]; }, src, dest);
	}

	template<class D>
	void zero(D & x) const {
		fordofs([&](len_t i) { x[i] = 0.; }, x);
	}

	template<class D>
	void set_random(D & x, unsigned seed) const {
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
	void set_to_scalar(scalar alpha, D & x) const {
		fordofs([&](len_t i) { x[i] = alpha; }, x);
	}

	template<class D>
	void scale(scalar alpha, D & x) const {
		fordofs([&](len_t i) { x[i] *= alpha; }, x);
	}

	template<class D0, class D1>
	void scale(scalar alpha, const D0 & x, D1 & y) const {
		fordofs([&](len_t i) { y[i] = alpha * x[i]; }, x, y);
	}

	template<class X, class Y, class Z>
	void add(const X & x, const Y & y, Z & z) const {
		fordofs([&](len_t i) { z[i] = x[i] + y[i]; }, x, y, z);
	}

	template<class X, class Y, class Z>
	void subtract(const X & x, const Y & y, Z & z) const {
		fordofs([&](len_t i) { z[i] = x[i] - y[i]; }, x, y, z);
	}

	template<class X, class Y, class Z>
	void multiply(const X & x, const Y & y, Z & z) const {
		fordofs([&](len_t i) { z[i] = x[i] * y[i]; }, x, y, z);
	}

	template<class X, class Y, class Z>
	void divide(const X & x, const Y & y, Z & z) const {
		fordofs([&](len_t i) { z[i] = x[i] / y[i]; }, x, y, z);
	}

	template<class X, class Y>
	void reciprocal(const X & x, Y & y) const {
		fordofs([&](len_t i) { x[i] = 1. / y[i]; }, x, y);
	}

	template<class X>
	void dump(std::string_view pre, const X & x) const {
		std::string fname{pre};
		fname += "-" + std::to_string(flecsi::color());
		std::ofstream ofile{fname};
		fordofs([&](len_t i) { ofile << x[i] << '\n'; }, x);
	}

	template<class X, class Y, class Z>
	void linear_sum(scalar alpha, const X & x, scalar beta, const Y & y, Z & z)
		const {
		fordofs([&](len_t i) { z[i] = alpha * x[i] + beta * y[i]; }, x, y, z);
	}

	template<class X, class Y, class Z>
	void axpy(scalar alpha, const X & x, const Y & y, Z & z) const {
		fordofs([&](len_t i) { z[i] = alpha * x[i] + y[i]; }, x, y, z);
	}

	template<class X, class Z>
	void axpby(scalar alpha, scalar beta, const X & x, Z & z) const {
		fordofs([&](len_t i) { z[i] = alpha * x[i] + beta * z[i]; }, x, z);
	}

	template<class X, class Y>
	void abs(const X & x, Y & y) const {
		fordofs([&](len_t i) { y[i] = std::abs(x[i]); }, x, y);
	}

	template<class X, class Y>
	void add_scalar(const X & x, scalar alpha, Y & y) const {
		fordofs([&](len_t i) { y[i] = x[i] + alpha; }, x, y);
	}

	template<class X>
	fut<real> min(const X & x) const {
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
	fut<real> max(const X & x) const {
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
	fut<real> lp_norm(const X & x) const {
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
		;
	}

	template<class X>
	fut<real> inf_norm(const X & x) const {
		auto ret = std::numeric_limits<real>::min();
		fordofs([&](len_t i) { ret = std::max(std::abs(x[i]), ret); }, x);
		return ret;
	}

	template<class X, class Y>
	fut<scalar> dot(const X & x, const Y & y) const {
		return scalar_prod(x, y);
	}

	template<class X>
	len_t local_size(const X & x) const {
		return x.data.size();
	}

	template<class X>
	fut<len_t> global_size(const X & x) const {
		return local_size(x);
	}

protected:
	template<class X, class Y>
	scalar scalar_prod(const X & x, const Y & y) const {
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
	void fordofs(F && f, const A0 & a0, const A &... arest) const {
		len_t sz = a0.size();
		([&](const auto & a) { flog_assert(a.size() == sz, ""); }(arest), ...);
		for (len_t i = 0; i < sz; ++i) {
			f(i);
		}
	}
};

template<class T>
struct seq_view : seq<seq_view<T>> {
	using base = seq<seq_view<T>>;
	using data_t = typename base::data_t;

	FLECSI_INLINE_TARGET seq_view(flecsi::util::span<T> span) : base{span} {}
};

template<class T, auto var = anon_var::anonymous>
struct seq_vec : seq<seq_vec<T, var>> {
	using base = seq<seq_vec<T, var>>;

	seq_vec() : base{} {}
	seq_vec(std::size_t sz) : base{sz} {}
};

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

namespace flecsolve {
template<class T>
struct traits<vec::seq_view<T>> {
	static constexpr auto var = variable<anon_var::anonymous>;
	using data_t = vec::span_view<T>;
	using ops_t = vec::seq_ops<typename data_t::scalar>;
};

template<class T, auto V>
struct traits<vec::seq_vec<T, V>> {
	static constexpr auto var = variable<V>;
	using data_t = std::vector<T>;
	using ops_t = vec::seq_ops<T>;
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
