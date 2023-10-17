#ifndef FLECSOLVE_OP_BASE_H
#define FLECSOLVE_OP_BASE_H

#include <functional>

#include "traits.hh"
#include "flecsolve/vectors/traits.hh"

namespace flecsolve::op {

template<class Derived>
struct base {
	using exact_type = Derived;
	static constexpr auto input_var = traits<Derived>::input_var;
	static constexpr auto output_var = traits<Derived>::output_var;
	using params_type = typename traits<Derived>::parameters;

	template<class T = params_type>
	base(T && p) : params(std::forward<T>(p)) {}

	template<class T = params_type,
	         class = std::enable_if_t<std::is_null_pointer_v<T>>>
	base() {}

	exact_type & derived() { return static_cast<Derived &>(*this); }

	const exact_type & derived() const {
		return static_cast<const Derived &>(*this);
	}

	template<class D,
	         class R,
	         std::enable_if_t<is_vector_v<D>, bool> = true,
	         std::enable_if_t<is_vector_v<R>, bool> = true>
	decltype(auto) apply(const D & x, R & y) const {
		return derived().apply(x, y);
	}

	template<class B,
	         class X,
	         class R,
	         std::enable_if_t<is_vector_v<B>, bool> = true,
	         std::enable_if_t<is_vector_v<X>, bool> = true,
	         std::enable_if_t<is_vector_v<R>, bool> = true>
	void residual(const B & b, const X & x, R & r) const {
		derived().residual_impl(b, x, r);
	}

	template<class B, class X, class R>
	void residual_impl(const B & b, const X & x, R & r) const {
		apply(x, r);

		const auto & bs = b.subset(output_var);
		auto & rs = r.subset(output_var);
		rs.subtract(bs, rs);
	}

	template<class T>
	void reset(const T & t) const {
		derived().reset_impl(t);
	}

	template<class T>
	void reset_impl(const T &) const {}

	auto & get_operator() { return derived().get_operator_impl(); }
	const auto & get_operator() const { return derived().get_operator_impl(); }
	exact_type & get_operator_impl() { return static_cast<Derived &>(*this); }
	const exact_type & get_operator_impl() const {
		return static_cast<const Derived &>(*this);
	}

	template<op::label tag, class T>
	auto get_parameters(const T & t) const {
		return derived().template get_parameters_impl<tag>(t);
	}

	template<op::label tag, class T>
	auto get_parameters_impl(const T &) const {
		return nullptr;
	}

protected:
	params_type params;
};

template<class T, class = void>
struct is_operator : std::false_type {};

template<class T>
struct is_operator<
	T,
	typename std::enable_if_t<
		std::is_base_of_v<base<std::decay_t<T>>, std::decay_t<T>>>>
	: std::true_type {};

template<class T>
inline constexpr bool is_operator_v = is_operator<T>::value;

}
#endif
