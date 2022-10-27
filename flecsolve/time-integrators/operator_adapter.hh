#ifndef FLECSI_LINALG_TIME_INTEGRATOR_OPERATOR_ADAPTER_H
#define FLECSI_LINALG_TIME_INTEGRATOR_OPERATOR_ADAPTER_H

#include <utility>

#include "flecsolve/vectors/base.hh"
#include "flecsolve/operators/base.hh"

namespace flecsolve::time_integrator {

template<class Op>
struct operator_adapter : op::base<operator_adapter<Op>> {
	template<class... Args>
	operator_adapter(Args &&... args)
		: op(std::forward<Args>(args)...), gamma{1.} {}

	template<class D, class R>
	void apply(const vec::base<D> & x, vec::base<R> & y) const {
		// f(x^{n+1})
		apply_rhs(x, y);
		// y = x^{n+1} - scaling * f(x^{n+1})
		y.axpy(-gamma, y, x);
	}

	template<class D, class R>
	void apply_rhs(const vec::base<D> & x, vec::base<R> & y) const {
		op.apply(x, y);
	}

	template<class V>
	bool is_valid(const vec::base<V> &) {
		return true;
	}

	double get_scaling() const { return gamma; }
	void set_scaling(double scaling) { gamma = scaling; }

protected:
	Op op;
	double gamma;
};

}

namespace flecsolve::op {

template<class Op>
struct traits<time_integrator::operator_adapter<Op>> {
	static constexpr auto input_var = Op::input_var;
	static constexpr auto output_var = Op::output_var;
	using parameters = std::nullptr_t;
};

}
#endif
