#pragma once
#include <utility>


namespace flecsi::linalg {

template <class Data, class Ops>
class vector
{
public:
	using vec = vector<Data, Ops>;
	using real_t = typename Ops::real_t;
	using len_t = typename Ops::len_t;
	using data_t = Data;
	using ops_t = Ops;

	vector(Data d) : data(std::move(d)) {}

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
	void zero() {
		return ops.zero(data);
	}


	/**
	 * Set vector components to scalar value.
	 *
	 * \f$\mathit{this}_i = val\f$
	 * \param[in] val scalar
	 */
	void set_to_scalar(real_t val) {
		ops.set_to_scalar(val, data);
	}


	/**
	 * Scale vector components.
	 *
	 * \f$\mathit{this}_i = alpha * x_i\f$
	 * \param[in] alpha scalar
	 * \param[in] x vector
	 */
	void scale(real_t alpha, const vec & x) {
		ops.scale(alpha, x.data,
		          data);
	}


	/**
	 * Scale vector components.
	 *
	 * \f$\mathit{this}_i = alpha * \mathit{this}_i\f$
	 * \param[in] alpha scalar
	 */
	void scale(real_t alpha) {
		ops.scale(alpha, data);
	}


	/**
	 * Component-wise addition of two vectors.
	 *
	 * \f$\mathit{this}_i = x_i + y_i\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	void add(const vec & x, const vec & y) {
		ops.add(x.data, y.data, data);
	}


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
	void reciprocal(const vec & x) {
		ops.reciprocal(x.data, data);
	}


	/**
	 * Set this to linear combination of two vectors.
	 *
	 * \f$\mathit{this}_i = alpha * x_i + beta * y_i\f$
	 */
	void linear_sum(real_t alpha, const vec & x, real_t beta,
	                const vec & y) {
		ops.linear_sum(alpha, x.data,
		               beta, y.data,
		               data);
	}


	/**
	 * Set this to alpha * x + y.
	 *
	 * \f$\mathit{this}_i = alpha x_i + y_i\f$
	 */
	void axpy(real_t alpha, const vec & x, const vec & y) {
		ops.axpy(alpha, x.data, y.data,
		         data);
	}


	/**
	 * Set this to alpha * x + beta * this.
	 *
	 * \f$\mathit{this}_i = alpha * x_i + beta * \mathit{this}_i\f$
	 */
	void axpby(real_t alpha, real_t beta, const vec & x) {
		ops.axpby(alpha, beta, x.data,
		          data);
	}


	/**
	 * Set this to component-wise absolute value of a vector
	 *
	 * \f$\mathit{this} = \abs{x_i}\f$
	 */
	void abs(const vec & x) {
		ops.abs(x.data, data);
	}


	/**
	 * Set this to the scalar translation of a vector
	 *
	 * \f$\mathit{this}_i = alpha x_i\f$
	 */
	void add_scalar(const vec & x, real_t alpha) {
		ops.add_scalar(x.data, alpha,
		               data);
	}


	/**
	 * Compute minimum value of vector.
	 *
	 * \f$ \min_i \mathit{this}_i/f$
	 * \return future containing the minimum value
	 */
	auto min() const {
		return ops.min(data);
	}


	/**
	 * Compute maximum value of vector.
	 *
	 * \f$ \max_i \mathit{this}_i/f$
	 * \return future containing the maximum value
	 */
	auto max() const {
		return ops.max(data);
	}


	/**
	 * Compute L_1 norm of this vector.
	 *
	 * \f$ \sum_i \abs{\mathit{this}_i} \f$
	 * \return future containing the L_1 norm
	 */
	auto l1norm() const {
		return ops.template lp_norm<1>(data);
	}


	/**
	 * Compute L_2 norm of this vector.
	 *
	 * \f$ \sqrt{\sum_i \left(\mathit{this}_i\right)^2} \f$
	 * \return future containing the L_2 norm
	 */
	auto l2norm() const {
		return ops.template lp_norm<2>(data);
	}


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
	auto inf_norm() const {
		return ops.inf_norm(data);
	}


	/**
	 * Compute inner product of two vectors
	 *
	 * \f$ \sum_i \mathit{this}_i x_i \f$
	 * \return future containing the inner product
	 */
	auto inner_prod(const vec & x) const {
		return ops.inner_prod(data, x.data);
	}


	/**
	 * Compute the global size of this vector.
	 *
	 * \return future containing the global size
	 */
	auto global_size() const {
		return ops.global_size(data);
	}


	/**
	 * Compute the local size of this vector.
	 *
	 * \return local size of this vector
	 */
	std::size_t local_size() const {
		return ops.local_size(data);
	}

	Data data;
	Ops ops;
};

}
