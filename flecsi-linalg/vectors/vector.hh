#pragma once
#include <utility>


namespace flecsi::linalg {

template <class Data, class Ops, class Real = double>
class vector
{
public:
	using vec = vector<Data, Ops, Real>;
	using real_t = Real;

	vector(Data && d) : data(std::forward<Data>(d)) {}

	void copy(const vec & other) {
		ops.copy(other.vector_data(), vector_data());
	}

	void zero() {
		return ops.zero(vector_data());
	}

	void set_to_scalar(real_t val) {
		ops.set_to_scalar(val, vector_data());
	}

	void scale(real_t alpha, const vec & x) {
		ops.scale(alpha, x.vector_data(),
		          vector_data());
	}

	void scale(real_t alpha) {
		ops.scale(alpha, vector_data());
	}

	void add(const vec & x, const vec & y) {
		ops.add(x.vector_data(), y.vector_data(), vector_data());
	}

	void subtract(const vec & x, const vec & y) {
		ops.subtract(x.vector_data(), y.vector_data(), vector_data());
	}

	void multiply(const vec & x, const vec & y) {
		ops.multiply(x.vector_data(), y.vector_data(), vector_data());
	}

	void divide(const vec & x, const vec & y) {
		ops.divide(x.vector_data(), y.vector_data(), vector_data());
	}

	void reciprocal(const vec & x) {
		ops.reciprocal(x.vector_data(), vector_data());
	}

	void linear_sum(real_t alpha, const vec & x, real_t beta,
	                const vec & y) {
		ops.linear_sum(alpha, x.vector_data(),
		               beta, y.vector_data(),
		               vector_data());
	}

	void axpy(real_t alpha, const vec & x, const vec & y) {
		ops.axpy(alpha, x.vector_data(), y.vector_data(),
		         vector_data());
	}

	void axpby(real_t alpha, real_t beta, const vec & x) {
		ops.axpby(alpha, beta, x.vector_data(),
		          vector_data());
	}

	void abs(const vec & x) {
		ops.abs(x.vector_data(), vector_data());
	}

	void add_scalar(const vec & x, real_t alpha) {
		ops.add_scalar(x.vector_data(), alpha,
		               vector_data());
	}

	auto min() const {
		return ops.min(vector_data());
	}

	auto max() const {
		return ops.max(vector_data());
	}

	auto l1norm() const {
		return ops.template lp_norm<1>(vector_data());
	}

	auto l2norm() const {
		return ops.template lp_norm<2>(vector_data());
	}

	template<unsigned short p>
	auto lp_norm() const {
		return ops.template lp_norm<p>(vector_data());
	}

	auto inf_norm() const {
		return ops.inf_norm(vector_data());
	}

	auto inner_prod(const vec & x) const {
		return ops.inner_prod(vector_data(), x.vector_data());
	}


	const Data & vector_data() const { return data; }
	Data & vector_data() { return data; }

	const Ops & vector_ops() const { return ops; }
	Ops & vector_ops() { return ops; }

protected:
	Data data;
	Ops ops;
};

}
