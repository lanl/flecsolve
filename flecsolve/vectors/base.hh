#ifndef FLECSOLVE_VECTORS_BASE_HH
#define FLECSOLVE_VECTORS_BASE_HH

#include <optional>
#include <random>
#include <type_traits>
#include <utility>

#include "flecsolve/util/traits.hh"
#include "variable.hh"

namespace flecsolve::vec {

template<class Derived>
struct base {
	using ops_t = typename traits<Derived>::ops_t;
	using data_t = typename traits<Derived>::data_t;

	using scalar = typename ops_t::scalar;
	using real = typename ops_t::real;
	using len_t = typename ops_t::len_t;

	static constexpr std::size_t num_components = 1;

	base() {}

	template<class D>
	base(D && d) : data(std::forward<D>(d)) {}

	Derived & derived() { return static_cast<Derived &>(*this); }

	const Derived & derived() const {
		return static_cast<const Derived &>(*this);
	}

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
	template<class T>
	void scale(scalar alpha, const T & x) {
		ops.scale(alpha, x.data, data);
	}

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
	template<class T0, class T1>
	void add(const T0 & x, const T1 & y) {
		ops.add(x.data, y.data, data);
	}

	/**
	 * Component-wise subtraction of two vectors.
	 *
	 * \f$\mathit{this}_i = x_i - y_i\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	template<class T0, class T1>
	void subtract(const T0 & x, const T1 & y) {
		ops.subtract(x.data, y.data, data);
	}

	/**
	 * Component-wise multiplication of two vectors.
	 *
	 * \f$\mathit{this}_i = x_i * y_i\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	template<class T0, class T1>
	void multiply(const T0 & x, const T1 & y) {
		ops.multiply(x.data, y.data, data);
	}

	/**
	 * Component-wise division of two vectors.
	 *
	 * \f$\mathit{this}_i = \frac{x_i}{y_i}\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	template<class T0, class T1>
	void divide(const T0 & x, const T1 & y) {
		ops.divide(x.data, y.data, data);
	}

	/**
	 * Set to the component-wise reciprocal of another vector.
	 *
	 * \f$\mathit{this} = 1.0 / x_i\f$
	 */
	template<class T>
	void reciprocal(const T & x) {
		ops.reciprocal(x.data, data);
	}

	/**
	 * Set this to linear combination of two vectors.
	 *
	 * \f$\mathit{this}_i = alpha * x_i + beta * y_i\f$
	 */
	template<class T0, class T1>
	void linear_sum(scalar alpha, const T0 & x, scalar beta, const T1 & y) {
		ops.linear_sum(alpha, x.data, beta, y.data, data);
	}

	/**
	 * Set this to alpha * x + y.
	 *
	 * \f$\mathit{this}_i = alpha x_i + y_i\f$
	 */
	template<class T0, class T1>
	void axpy(scalar alpha, const T0 & x, const T1 & y) {
		ops.axpy(alpha, x.data, y.data, data);
	}

	/**
	 * Set this to alpha * x + beta * this.
	 *
	 * \f$\mathit{this}_i = alpha * x_i + beta * \mathit{this}_i\f$
	 */
	template<class T>
	void axpby(scalar alpha, scalar beta, const T & x) {
		ops.axpby(alpha, beta, x.data, data);
	}

	/**
	 * Set this to component-wise absolute value of a vector
	 *
	 * \f$\mathit{this} = \abs{x_i}\f$
	 */
	template<class T>
	void abs(const T & x) {
		ops.abs(x.data, data);
	}

	/**
	 * Set this to the scalar translation of a vector
	 *
	 * \f$\mathit{this}_i = alpha x_i\f$
	 */
	template<class T>
	void add_scalar(const T & x, scalar alpha) {
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
	template<class T>
	auto dot(const T & x) const {
		return ops.dot(data, x.data);
	}

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

	template<class T>
	constexpr decltype(auto) subset(T && t) const {
		return derived().subset_impl(std::forward<T>(t));
	}

	template<class T>
	constexpr decltype(auto) subset(T && t) {
		return derived().subset_impl(std::forward<T>(t));
	}

	template<auto ovar>
	constexpr decltype(auto) subset_impl(variable_t<ovar>) const {
		static_assert(ovar == var.value);
		return *this;
	}

	template<auto ovar>
	constexpr decltype(auto) subset_impl(variable_t<ovar>) {
		static_assert(ovar == var.value);
		return *this;
	}

	template<auto ovar>
	constexpr decltype(auto) subset_impl(multivariable_t<ovar>) const {
		static_assert(ovar == var.value);
		return *this;
	}

	template<auto ovar>
	constexpr decltype(auto) subset_impl(multivariable_t<ovar>) {
		static_assert(ovar == var.value);
		return *this;
	}

	template<class F>
	constexpr decltype(auto) apply(F && f) {
		return derived().apply_impl(std::forward<F>(f));
	}

	template<class F>
	constexpr decltype(auto) apply_impl(F && f) {
		return std::forward<F>(f)(*this);
	}

	data_t data;
	ops_t ops;
	static constexpr auto var = traits<Derived>::var;
	// using var_t =
	// std::conditional_t<is_variable_v<std::remove_const_t<decltype(var)>>,
	// 	std::remove_const_t<decltype(var.value)>, std::nullptr_t>;
	using var_t = std::remove_const_t<decltype(var.value)>;
};

template<class Derived>
bool operator==(const base<Derived> & v1, const base<Derived> & v2) {
	return v1.data == v2.data;
}

template<class Derived>
bool operator!=(const base<Derived> & v1, const base<Derived> & v2) {
	return v1.data != v2.data;
}

}
#endif
