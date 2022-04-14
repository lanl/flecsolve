#pragma once

#include <optional>
#include <random>
#include <type_traits>
#include <utility>

#include "variable.hh"

namespace flecsi::linalg {

template<class Data, class Ops, auto Variable = anon_var::anonymous>
class vector
{
public:
	using vec = vector<Data, Ops, Variable>;

	using scalar = typename Ops::scalar;
	using real = typename Ops::real;
	using len_t = typename Ops::len_t;

	using data_t = Data;
	using ops_t = Ops;

	template<class D>
	vector(D && d) : data(std::forward<D>(d)) {}

	/**
	 * Set vector equal to other.
	 *
	 * \f$\mathit{this}_i = \mathit{other}_i\f$
	 * \param[in] other vector
	 */
	template<class Other>
	void copy(const Other & other) {
		ops.copy(other.data, data);
	}

	/**
	 * Set vector components to 0.
	 */
	void zero() { return ops.zero(data); }

	/**
	 * Set vector components to scalar value.
	 *
	 * \f$\mathit{this}_i = val\f$
	 * \param[in] val scalar
	 */
	void set_scalar(scalar val) { ops.set_to_scalar(val, data); }

	/**
	 * Scale vector components.
	 *
	 * \f$\mathit{this}_i = alpha * x_i\f$
	 * \param[in] alpha scalar
	 * \param[in] x vector
	 */
	void scale(scalar alpha, const vec & x) { ops.scale(alpha, x.data, data); }

	/**
	 * Scale vector components.
	 *
	 * \f$\mathit{this}_i = alpha * \mathit{this}_i\f$
	 * \param[in] alpha scalar
	 */
	void scale(scalar alpha) { ops.scale(alpha, data); }

	/**
	 * Component-wise addition of two vectors.
	 *
	 * \f$\mathit{this}_i = x_i + y_i\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	void add(const vec & x, const vec & y) { ops.add(x.data, y.data, data); }

	/**
	 * Component-wise subtraction of two vectors.
	 *
	 * \f$\mathit{this}_i = x_i - y_i\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	void subtract(const vec & x, const vec & y) {
		ops.subtract(x.data, y.data, data);
	}

	/**
	 * Component-wise multiplication of two vectors.
	 *
	 * \f$\mathit{this}_i = x_i * y_i\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	void multiply(const vec & x, const vec & y) {
		ops.multiply(x.data, y.data, data);
	}

	/**
	 * Component-wise division of two vectors.
	 *
	 * \f$\mathit{this}_i = \frac{x_i}{y_i}\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	void divide(const vec & x, const vec & y) {
		ops.divide(x.data, y.data, data);
	}

	/**
	 * Set to the component-wise reciprocal of another vector.
	 *
	 * \f$\mathit{this} = 1.0 / x_i\f$
	 */
	void reciprocal(const vec & x) { ops.reciprocal(x.data, data); }

	/**
	 * Set this to linear combination of two vectors.
	 *
	 * \f$\mathit{this}_i = alpha * x_i + beta * y_i\f$
	 */
	void linear_sum(scalar alpha, const vec & x, scalar beta, const vec & y) {
		ops.linear_sum(alpha, x.data, beta, y.data, data);
	}

	/**
	 * Set this to alpha * x + y.
	 *
	 * \f$\mathit{this}_i = alpha x_i + y_i\f$
	 */
	void axpy(scalar alpha, const vec & x, const vec & y) {
		ops.axpy(alpha, x.data, y.data, data);
	}

	/**
	 * Set this to alpha * x + beta * this.
	 *
	 * \f$\mathit{this}_i = alpha * x_i + beta * \mathit{this}_i\f$
	 */
	void axpby(scalar alpha, scalar beta, const vec & x) {
		ops.axpby(alpha, beta, x.data, data);
	}

	/**
	 * Set this to component-wise absolute value of a vector
	 *
	 * \f$\mathit{this} = \abs{x_i}\f$
	 */
	void abs(const vec & x) { ops.abs(x.data, data); }

	/**
	 * Set this to the scalar translation of a vector
	 *
	 * \f$\mathit{this}_i = alpha x_i\f$
	 */
	void add_scalar(const vec & x, scalar alpha) {
		ops.add_scalar(x.data, alpha, data);
	}

	/**
	 * Compute minimum value of vector.
	 *
	 * \f$ \min_i \mathit{this}_i/f$
	 * \return future containing the minimum value
	 */
	auto min() const { return ops.min(data); }

	/**
	 * Compute maximum value of vector.
	 *
	 * \f$ \max_i \mathit{this}_i/f$
	 * \return future containing the maximum value
	 */
	auto max() const { return ops.max(data); }

	/**
	 * Compute L_1 norm of this vector.
	 *
	 * \f$ \sum_i \abs{\mathit{this}_i} \f$
	 * \return future containing the L_1 norm
	 */
	auto l1norm() const { return ops.template lp_norm<1>(data); }

	/**
	 * Compute L_2 norm of this vector.
	 *
	 * \f$ \sqrt{\sum_i \left(\mathit{this}_i\right)^2} \f$
	 * \return future containing the L_2 norm
	 */
	auto l2norm() const { return ops.template lp_norm<2>(data); }

	/**
	 * Compute L_p norm of this vector.
	 *
	 * \return future containing the L_p norm
	 */
	template<unsigned short p>
	auto lp_norm() const {
		return ops.template lp_norm<p>(data);
	}

	/**
	 * Compute L_\infty norm of this vector.
	 *
	 * \f$ \max_i \abs{\mathit{this}_i} \f$
	 */
	auto inf_norm() const { return ops.inf_norm(data); }

	/**
	 * Compute inner product of two vectors
	 *
	 * \f$ \sum_i \mathit{this}_i x_i \f$
	 * for complex numbers:
	 * \f$ \sum_i x_i^H \mathit{this}_i \f$
	 * \return future containing the inner product
	 */
	auto dot(const vec & x) const { return ops.dot(data, x.data); }

	/**
	 * Compute the global size of this vector.
	 *
	 * \return future containing the global size
	 */
	auto global_size() const { return ops.global_size(data); }

	/**
	 * Compute the local size of this vector.
	 *
	 * \return local size of this vector
	 */
	std::size_t local_size() const { return ops.local_size(data); }

	/**
	 * Set components to random values.
	 */
	void set_random() {
		std::random_device rd;
		ops.set_random(data, rd());
	}

	/**
	 * Set components to random values given a seed.
	 */
	void set_random(unsigned seed) { ops.set_random(data, seed); }

	void dump(std::string_view str) { ops.dump(str, data); }

	template<auto var>
	constexpr decltype(auto) subset(variable_t<var>) {
		static_assert(var == Variable);
		return *this;
	}

	template<auto var>
	constexpr decltype(auto) subset(multivariable_t<var>) {
		static_assert(var == Variable);
		return *this;
	}

	Data data;
	Ops ops;
	static constexpr auto var = Variable;
	using var_t = decltype(var);
};

} // namespace flecsi::linalg
