#pragma once
#include <utility>


namespace flecsi::linalg {

template <class Data, class Ops, class Real = double>
class vector
{
public:
	using vec = vector<Data, Ops, Real>;
	using real_t = Real;
	using data_t = Data;
	using ops_t = Ops;

	vector(Data d) : data(std::move(d)) {}

	void copy(const vec & other) {
		ops.copy(other.data, data);
	}

	void zero() {
		return ops.zero(data);
	}

	void set_to_scalar(real_t val) {
		ops.set_to_scalar(val, data);
	}

	void scale(real_t alpha, const vec & x) {
		ops.scale(alpha, x.data,
		          data);
	}

	void scale(real_t alpha) {
		ops.scale(alpha, data);
	}

	void add(const vec & x, const vec & y) {
		ops.add(x.data, y.data, data);
	}

	void subtract(const vec & x, const vec & y) {
		ops.subtract(x.data, y.data, data);
	}

	void multiply(const vec & x, const vec & y) {
		ops.multiply(x.data, y.data, data);
	}

	void divide(const vec & x, const vec & y) {
		ops.divide(x.data, y.data, data);
	}

	void reciprocal(const vec & x) {
		ops.reciprocal(x.data, data);
	}

	void linear_sum(real_t alpha, const vec & x, real_t beta,
	                const vec & y) {
		ops.linear_sum(alpha, x.data,
		               beta, y.data,
		               data);
	}

	void axpy(real_t alpha, const vec & x, const vec & y) {
		ops.axpy(alpha, x.data, y.data,
		         data);
	}

	void axpby(real_t alpha, real_t beta, const vec & x) {
		ops.axpby(alpha, beta, x.data,
		          data);
	}

	void abs(const vec & x) {
		ops.abs(x.data, data);
	}

	void add_scalar(const vec & x, real_t alpha) {
		ops.add_scalar(x.data, alpha,
		               data);
	}

	auto min() const {
		return ops.min(data);
	}

	auto max() const {
		return ops.max(data);
	}

	auto l1norm() const {
		return ops.template lp_norm<1>(data);
	}

	auto l2norm() const {
		return ops.template lp_norm<2>(data);
	}

	template<unsigned short p>
	auto lp_norm() const {
		return ops.template lp_norm<p>(data);
	}

	auto inf_norm() const {
		return ops.inf_norm(data);
	}

	auto inner_prod(const vec & x) const {
		return ops.inner_prod(data, x.data);
	}


	Data data;
	Ops ops;
};

}
