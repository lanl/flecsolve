#ifndef FLECSI_LINALG_TIME_INTEGRATOR_OPERATOR_ADAPTER_H
#define FLECSI_LINALG_TIME_INTEGRATOR_OPERATOR_ADAPTER_H

#include <utility>

namespace flecsolve::time_integrator {

template<class Op>
struct operator_adapter : Op {
	template<class... Args>
	operator_adapter(Args &&... args)
		: Op(std::forward<Args>(args)...), gamma{1.} {}

	template<class D, class R>
	void apply(const D & x, R & y) const {
		// f(x^{n+1})
		apply_rhs(x, y);
		// y = x^{n+1} - scaling * f(x^{n+1})
		y.axpy(-gamma, y, x);
	}

	template<class D, class R>
	decltype(auto) apply_rhs(const D & x, R & y) const {
		return static_cast<const Op &>(*this).apply(x, y);
	}

	template<class V>
	bool is_valid(const V &) {
		return true;
	}

	double get_scaling() const { return gamma; }
	void set_scaling(double scaling) { gamma = scaling; }

protected:
	double gamma;
};

}

#endif
