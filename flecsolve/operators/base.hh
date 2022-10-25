#ifndef FLECSOLVE_OP_BASE_H
#define FLECSOLVE_OP_BASE_H

#include "traits.hh"

namespace flecsolve::op {

template<class Derived>
struct base {
	static constexpr auto input_var = traits<Derived>::input_var;
	static constexpr auto output_var = traits<Derived>::output_var;

	Derived & derived() { return static_cast<Derived &>(*this); }

	const Derived & derived() const {
		return static_cast<const Derived &>(*this);
	}

	template<class D, class R>
	decltype(auto) apply(const vec::base<D> & x, vec::base<R> & y) const {
		return derived().apply(x.derived(), y.derived());
	}

	template<class D, class R>
	void residual(const vec::base<R> & b,
	              const vec::base<D> & x,
	              vec::base<R> & r) const {
		if constexpr (has_residual_v<Derived, D, R>) {
			derived().residual(b.derived(), x.derived(), r.derived());
		}
		else {
			apply(x.derived(), r.derived());

			const auto & bs = b.subset(output_var);
			auto & rs = r.subset(output_var);
			rs.subtract(bs.derived(), rs.derived());
		}
	}

	template<class T>
	void reset(const T & t) const {
		derived().reset_impl(t);
	}

	template<class T>
	void reset_impl(const T &) const {}

	auto & get_operator() { return derived().get_operator_impl(); }
	const auto & get_operator() const { return derived().get_operator_impl(); }
	auto & get_operator_impl() { return *this; }
	const auto & get_operator_impl() const { return *this; }
};

}
#endif
