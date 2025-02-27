/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
#ifndef FLECSOLVE_VECTORS_CORE_HH
#define FLECSOLVE_VECTORS_CORE_HH

#include <random>

#include "flecsolve/util/traits.hh"
#include "variable.hh"

namespace flecsolve::vec {

template<template<class> class Data, template<class> class Ops, class Config>
struct core : Config {
	using ops = Ops<Data<Config>>;
	using data_t = Data<Config>;
	using scalar = typename Config::scalar;
	using len_t = typename Config::len_t;
	using config = Config;

	template<class T>
	class is_vector
	{
		template<template<class> class D1, template<class> class O1, class C1>
		static decltype(static_cast<core<D1, O1, C1>>(std::declval<T>()),
		                std::true_type{}) test(const core<D1, O1, C1> &);
		static std::false_type test(...);

	public:
		static constexpr bool value =
			decltype(is_vector::test(std::declval<T>()))::value;
	};
	template<class T>
	static constexpr bool is_vector_v = is_vector<T>::value;
	static constexpr auto var = Config::var;
	using var_t = typename Config::var_t;

	explicit core(data_t d) : data(std::move(d)) {}

	/**
	 * Set vector equal to other.
	 *
	 * \f$\mathit{this}_i = \mathit{other}_i\f$
	 * \param[in] other vector
	 */
	template<class V, std::enable_if_t<is_vector_v<V>, bool> = true>
	void copy(const V & src) {
		ops::copy(src.data, data);
	}

	/**
	 * Set vector components to 0.
	 */
	void zero() { return ops::zero(data); }

	/**
	 * Set vector components to scalar value.
	 *
	 * \f$\mathit{this}_i = val\f$
	 * \param[in] val scalar
	 */
	void set_scalar(scalar val) { ops::set_to_scalar(val, data); }

	/**
	 * Scale vector components.
	 *
	 * \f$\mathit{this}_i = alpha * x_i\f$
	 * \param[in] alpha scalar
	 * \param[in] x vector
	 */
	template<class V, std::enable_if_t<is_vector_v<V>, bool> = true>
	void scale(scalar alpha, const V & x) {
		ops::scale(alpha, x.data, data);
	}

	/**
	 * Scale vector components.
	 *
	 * \f$\mathit{this}_i = alpha * \mathit{this}_i\f$
	 * \param[in] alpha scalar
	 */
	void scale(scalar alpha) { ops::scale(alpha, data); }

	/**
	 * Component-wise addition of two vectors.
	 *
	 * \f$\mathit{this}_i = x_i + y_i\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	template<class V1,
	         class V2,
	         std::enable_if_t<is_vector_v<V1>, bool> = true,
	         std::enable_if_t<is_vector_v<V2>, bool> = true>
	void add(const V1 & x, const V2 & y) {
		ops::add(x.data, y.data, data);
	}

	/**
	 * Component-wise subtraction of two vectors.
	 *
	 * \f$\mathit{this}_i = x_i - y_i\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	template<class V1,
	         class V2,
	         std::enable_if_t<is_vector_v<V1>, bool> = true,
	         std::enable_if_t<is_vector_v<V2>, bool> = true>
	void subtract(const V1 & x, const V2 & y) {
		ops::subtract(x.data, y.data, data);
	}

	/**
	 * Component-wise multiplication of two vectors.
	 *
	 * \f$\mathit{this}_i = x_i * y_i\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	template<class V1,
	         class V2,
	         std::enable_if_t<is_vector_v<V1>, bool> = true,
	         std::enable_if_t<is_vector_v<V2>, bool> = true>
	void multiply(const V1 & x, const V2 & y) {
		ops::multiply(x.data, y.data, data);
	}

	/**
	 * Component-wise division of two vectors.
	 *
	 * \f$\mathit{this}_i = \frac{x_i}{y_i}\f$
	 * \param[in] x vector
	 * \param[in] y vector
	 */
	template<class V1,
	         class V2,
	         std::enable_if_t<is_vector_v<V1>, bool> = true,
	         std::enable_if_t<is_vector_v<V2>, bool> = true>
	void divide(const V1 & x, const V2 & y) {
		ops::divide(x.data, y.data, data);
	}

	/**
	 * Set to the component-wise reciprocal of another vector.
	 *
	 * \f$\mathit{this} = 1.0 / x_i\f$
	 */
	template<class V, std::enable_if_t<is_vector_v<V>, bool> = true>
	void reciprocal(const V & x) {
		ops::reciprocal(x.data, data);
	}

	/**
	 * Set this to linear combination of two vectors.
	 *
	 * \f$\mathit{this}_i = alpha * x_i + beta * y_i\f$
	 */
	template<class V1,
	         class V2,
	         std::enable_if_t<is_vector_v<V1>, bool> = true,
	         std::enable_if_t<is_vector_v<V2>, bool> = true>
	void linear_sum(scalar alpha, const V1 & x, scalar beta, const V2 & y) {
		ops::linear_sum(alpha, x.data, beta, y.data, data);
	}

	/**
	 * Set this to alpha * x + y.
	 *
	 * \f$\mathit{this}_i = alpha x_i + y_i\f$
	 */
	template<class V1,
	         class V2,
	         std::enable_if_t<is_vector_v<V1>, bool> = true,
	         std::enable_if_t<is_vector_v<V2>, bool> = true>
	void axpy(scalar alpha, const V1 & x, const V2 & y) {
		ops::axpy(alpha, x.data, y.data, data);
	}

	/**
	 * Set this to alpha * x + beta * this.
	 *
	 * \f$\mathit{this}_i = alpha * x_i + beta * \mathit{this}_i\f$
	 */
	template<class V, std::enable_if_t<is_vector_v<V>, bool> = true>
	void axpby(scalar alpha, scalar beta, const V & x) {
		ops::axpby(alpha, beta, x.data, data);
	}

	/**
	 * Set this to component-wise absolute value of a vector
	 *
	 * \f$\mathit{this} = \abs{x_i}\f$
	 */
	template<class V, std::enable_if_t<is_vector_v<V>, bool> = true>
	void abs(const V & x) {
		ops::abs(x.data, data);
	}

	/**
	 * Set this to the scalar translation of a vector
	 *
	 * \f$\mathit{this}_i = alpha x_i\f$
	 */
	template<class V, std::enable_if_t<is_vector_v<V>, bool> = true>
	void add_scalar(const V & x, scalar alpha) {
		ops::add_scalar(x.data, alpha, data);
	}

	/**
	 * Compute minimum value of vector.
	 *
	 * \f$ \min_i \mathit{this}_i/f$
	 * \return future containing the minimum value
	 */
	auto min() const { return ops::min(data); }

	/**
	 * Compute maximum value of vector.
	 *
	 * \f$ \max_i \mathit{this}_i/f$
	 * \return future containing the maximum value
	 */
	auto max() const { return ops::max(data); }

	/**
	 * Compute L_1 norm of this vector.
	 *
	 * \f$ \sum_i \abs{\mathit{this}_i} \f$
	 * \return future containing the L_1 norm
	 */
	auto l1norm() const { return ops::template lp_norm<1>(data); }

	/**
	 * Compute L_2 norm of this vector.
	 *
	 * \f$ \sqrt{\sum_i \left(\mathit{this}_i\right)^2} \f$
	 * \return future containing the L_2 norm
	 */
	auto l2norm() const { return ops::template lp_norm<2>(data); }

	/**
	 * Compute L_p norm of this vector.
	 *
	 * \return future containing the L_p norm
	 */
	template<unsigned short p>
	auto lp_norm() const {
		return ops::template lp_norm<p>(data);
	}

	/**
	 * Compute L_\infty norm of this vector.
	 *
	 * \f$ \max_i \abs{\mathit{this}_i} \f$
	 */
	auto inf_norm() const { return ops::inf_norm(data); }

	/**
	 * Compute inner product of two vectors
	 *
	 * \f$ \sum_i \mathit{this}_i x_i \f$
	 * for complex numbers:
	 * \f$ \sum_i x_i^H \mathit{this}_i \f$
	 * \return future containing the inner product
	 */
	template<class V, std::enable_if_t<is_vector_v<V>, bool> = true>
	auto dot(const V & x) const {
		return ops::dot(data, x.data);
	}

	/**
	 * Compute the global size of this vector.
	 *
	 * \return future containing the global size
	 */
	auto global_size() const { return ops::global_size(data); }

	/**
	 * Compute the local size of this vector.
	 *
	 * \return local size of this vector
	 */
	std::size_t local_size() const { return ops::local_size(data); }

	/**
	 * Set components to random values.
	 */
	void set_random() {
		std::random_device rd;
		ops::set_random(data, rd());
	}

	/**
	 * Set components to random values given a seed.
	 */
	void set_random(unsigned seed) { ops::set_random(data, seed); }

	void dump(std::string_view str) { ops::dump(str, data); }

	template<auto ovar>
	constexpr decltype(auto) subset(variable_t<ovar>) const {
		static_assert(ovar == var.value);
		return *this;
	}

	template<auto ovar>
	constexpr decltype(auto) subset(variable_t<ovar>) {
		static_assert(ovar == var.value);
		return *this;
	}

	template<auto ovar>
	constexpr decltype(auto) subset(multivariable_t<ovar>) const {
		static_assert(ovar == var.value);
		return *this;
	}

	template<auto ovar>
	constexpr decltype(auto) subset(multivariable_t<ovar>) {
		static_assert(ovar == var.value);
		return *this;
	}

	scalar & operator[](len_t i) { return ops::retreive(data, i); }

	const scalar & operator[](len_t i) const { return ops::retreive(data, i); }

	template<class F>
	constexpr decltype(auto) apply(F && f) {
		if constexpr (config::num_components == 1) {
			return std::forward<F>(f)(*this);
		}
		else {
			return std::apply(std::forward<F>(f), data);
		}
	}

	data_t data;
};

template<template<class> class Data, template<class> class Ops, class Config>
bool operator==(const core<Data, Ops, Config> & v1,
                const core<Data, Ops, Config> & v2) {
	return v1.data == v2.data;
}

template<template<class> class Data, template<class> class Ops, class Config>
bool operator!=(const core<Data, Ops, Config> & v1,
                const core<Data, Ops, Config> & v2) {
	return v1.data != v2.data;
}

}

#endif
