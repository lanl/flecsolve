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
#ifndef FLECSOLVE_TIME_INTEGRATOR_FACTORY_H
#define FLECSOLVE_TIME_INTEGRATOR_FACTORY_H

#include <istream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "flecsolve/operators/factory.hh"
#include "flecsolve/time-integrators/bdf.hh"
#include "flecsolve/time-integrators/rk23.hh"
#include "flecsolve/time-integrators/rk45.hh"

namespace flecsolve::time_integrator {

enum class method_target { rk23, rk45, bdf };

inline std::istream & operator>>(std::istream & in, method_target & target) {
	std::string token;
	in >> token;
	if (token == "rk23")
		target = method_target::rk23;
	else if (token == "rk45")
		target = method_target::rk45;
	else if (token == "bdf")
		target = method_target::bdf;
	else
		in.setstate(std::ios_base::failbit);
	return in;
}

// A shared time operator applies the implicit system through apply(), and
// exposes the explicit RHS through apply_rhs(). Preserve its handle ownership.
template<class O>
struct time_op_rhs : op::base<> {
	static constexpr auto input_var = O::input_var;
	static constexpr auto output_var = O::output_var;
	explicit time_op_rhs(op::handle<O> time_op) : time_op(std::move(time_op)) {}

	template<class D, class R>
	void apply(const D & x, R & y) const {
		time_op.get().apply_rhs(x, y);
	}

private:
	op::handle<O> time_op;
};

template<method_target T>
struct rk_method {
};

template<>
struct rk_method<method_target::rk23> {
	using settings = rk23::settings;
	using options = rk23::options;

	template<class V, class O>
	static auto
	make_integrator(const settings & s, const V & v, op::handle<O> rhs) {
		return rk23::integrator(
			rk23::parameters(s, std::move(rhs), rk23::make_work(v)));
	}

	template<class V, class O>
	static auto make_stepper(const V & v, op::handle<O> rhs) {
		return rk23::stepper(
			rk23::parameters(std::move(rhs), rk23::make_work(v)));
	}
};

template<>
struct rk_method<method_target::rk45> {
	using settings = rk45::settings;
	using options = rk45::options;

	template<class V, class O>
	static auto
	make_integrator(const settings & s, const V & v, op::handle<O> rhs) {
		return rk45::integrator(
			rk45::parameters(s, std::move(rhs), rk45::make_work(v)));
	}

	template<class V, class O>
	static auto make_stepper(const V & v, op::handle<O> rhs) {
		return rk45::stepper(
			rk45::parameters(std::move(rhs), rk45::make_work(v)));
	}
};


template<method_target T>
struct integrator_registry {
	using rk = rk_method<T>;
	using settings = typename rk::settings;
	using options = typename rk::options;

	template<class V, class O>
	static auto make(const settings & s, const V & v, op::handle<O> rhs) {
		return rk::make_integrator(s, v, std::move(rhs));
	}

	template<class V, class O, class S>
	static auto make(const settings & s,
	                 const V & v,
	                 op::handle<O> time_op,
	                 op::handle<S>) {
		return make(s, v, op::make_shared<time_op_rhs<O>>(std::move(time_op)));
	}
};


template<>
struct integrator_registry<method_target::bdf> {
	using settings = bdf::settings;
	using options = bdf::options;

	template<class V, class O, class S>
	static auto make(const settings & s,
	                 const V & v,
	                 op::handle<O> time_op,
	                 op::handle<S> solver) {
		return bdf::integrator(
			bdf::parameters(s, time_op, bdf::make_work(v), solver));
	}
};

// Method-specific empty settings keep the configuration variant unambiguous:
// RK23 and RK45 share the same underlying stepper_settings type.
template<method_target T>
struct stepper_registry {
	using rk = rk_method<T>;
	using settings = null_settings<T>;
	using options = null_options<T>;

	template<class V, class O>
	static auto make(const settings &, const V & v, op::handle<O> rhs) {
		return rk::make_stepper(v, std::move(rhs));
	}

	template<class V, class O, class S>
	static auto make(const settings & s,
	                 const V & v,
	                 op::handle<O> time_op,
	                 op::handle<S>) {
		return make(s, v, op::make_shared<time_op_rhs<O>>(std::move(time_op)));
	}
};

template<>
struct stepper_registry<method_target::bdf> {
	using settings = bdf::stepper_settings;
	using options = bdf::stepper_options;

	template<class V, class O, class S>
	static auto make(const settings & s,
	                 const V & v,
	                 op::handle<O> time_op,
	                 op::handle<S> solver) {
		return bdf::stepper(
			bdf::parameters(s, time_op, bdf::make_work(v), solver));
	}
};

template<template<method_target> class Registry, method_target... T>
struct method_policy {
	using target = method_target;
	using targets = includes<T...>;

	template<target V>
	using registry = Registry<V>;
};

template<class T>
struct is_bdf_stepper : std::false_type {
};

template<class O, class W, class S, class Settings>
struct is_bdf_stepper<bdf::stepper<O, W, S, Settings>> : std::true_type {
};

template<class Variant>
struct stepper_product {
	explicit stepper_product(Variant v) : var(std::move(v)) {}

	template<class DeltaT,
	         class Curr,
	         class Out,
	         std::enable_if_t<is_scalar_or_future_v<std::decay_t<DeltaT>,
	                                                typename Curr::scalar>,
	                          bool> = true>
	void advance(DeltaT dt, Curr & curr, Out & out) {
		std::visit(
			[&](auto & stepper) {
				if constexpr (is_bdf_stepper<
								  std::decay_t<decltype(stepper)>>::value &&
			                  !std::is_convertible_v<DeltaT, double>)
					throw std::invalid_argument(
						"BDF stepper requires a scalar timestep");
				else
					stepper.advance(std::move(dt), curr, out);
			},
			var);
	}

	bool check_solution() {
		return std::visit(
			[](auto & stepper) {
				if constexpr (is_bdf_stepper<
								  std::decay_t<decltype(stepper)>>::value)
					return stepper.check_solution();
				else
					return true;
			},
			var);
	}

	void update() {
		std::visit(
			[](auto & stepper) {
				if constexpr (is_bdf_stepper<
								  std::decay_t<decltype(stepper)>>::value)
					stepper.update();
			},
			var);
		++accepted_steps;
	}

	int get_current_step() const { return accepted_steps; }

	int num_step_rejects() const {
		return std::visit(
			[](const auto & stepper) {
				if constexpr (is_bdf_stepper<
								  std::decay_t<decltype(stepper)>>::value)
					return stepper.num_step_rejects();
				else
					return 0;
			},
			var);
	}

private:
	Variant var;
	int accepted_steps{0};
};


template<class Variant>
struct integrator_product {
	explicit integrator_product(Variant v) : var(std::move(v)) {}

	template<class Curr, class Out>
	void advance(double dt, Curr & curr, Out & out) {
		std::visit([&](auto & ti) { ti.advance(dt, curr, out); }, var);
	}

	bool check_solution() {
		return std::visit([](auto & ti) { return ti.check_solution(); }, var);
	}

	void update() {
		std::visit([](auto & ti) { ti.update(); }, var);
	}

	double get_next_dt(bool good_solution) {
		return std::visit(
			[&](auto & ti) { return ti.get_next_dt(good_solution); }, var);
	}

	double get_current_time() const {
		return std::visit([](const auto & ti) { return ti.get_current_time(); },
		                  var);
	}

	double get_final_time() const {
		return std::visit([](const auto & ti) { return ti.get_final_time(); },
		                  var);
	}

	double get_current_dt() const {
		return std::visit([](const auto & ti) { return ti.get_current_dt(); },
		                  var);
	}

	int get_current_step() const {
		return std::visit([](const auto & ti) { return ti.get_current_step(); },
		                  var);
	}

	int num_step_rejects() const {
		return std::visit([](const auto & ti) { return ti.num_step_rejects(); },
		                  var);
	}

	bool fixed_dt() const {
		return std::visit([](const auto & ti) { return ti.fixed_dt(); }, var);
	}

	bool steps_remaining() const {
		return std::visit([](const auto & ti) { return ti.steps_remaining(); },
		                  var);
	}

private:
	Variant var;
};

template<class T, class Variant>
struct is_variant_alternative : std::false_type {
};

template<class T, class... A>
struct is_variant_alternative<T, std::variant<A...>>
	: std::disjunction<std::is_same<T, A>...> {
};


template<template<method_target> class Registry, template<class> class Product>
struct method_factory {
	using implementation = op::factory<method_policy<Registry,
	                                                 method_target::rk23,
	                                                 method_target::rk45,
	                                                 method_target::bdf>>;
	using target = typename implementation::target;
	using settings = typename implementation::settings;
	using options = typename implementation::options;

	template<class V, class O>
	static auto make(const settings & s, const V & v, op::handle<O> rhs) {
		using rk_factory = op::factory<
			method_policy<Registry, method_target::rk23, method_target::rk45>>;
		using rk_types = typename rk_factory::settings_types;
		if (s.target_id.value() == target::bdf)
			throw std::invalid_argument(
				"BDF requires a time operator and solver handle");
		typename rk_factory::settings rk_settings{
			s.target_id,
			std::visit(
				[](const auto & ts) -> rk_types {
					using T = std::decay_t<decltype(ts)>;
					if constexpr (is_variant_alternative<T, rk_types>::value)
						return rk_types(std::in_place_type<T>, ts);
					else
						throw std::invalid_argument(
							"BDF requires a time operator and solver handle");
				},
				s.target_settings)};
		auto var = rk_factory::make_policy(rk_settings, v, std::move(rhs));
		return Product<decltype(var)>(std::move(var));
	}

	template<class V, class O, class S>
	static auto make(const settings & s,
	                 const V & v,
	                 op::handle<O> time_op,
	                 op::handle<S> solver) {
		auto var = implementation::make_policy(
			s, v, std::move(time_op), std::move(solver));
		return Product<decltype(var)>(std::move(var));
	}
};

struct stepper_factory : method_factory<stepper_registry, stepper_product> {
	using method_factory::make;

	template<class V, class O>
	static auto make(target method, const V & v, op::handle<O> rhs) {
		if (method == target::bdf)
			throw std::invalid_argument(
				"BDF stepper requires method settings and a solver handle");
		return make(settings{method, implementation::make_settings(method)},
		            v,
		            std::move(rhs));
	}
};

struct integrator_factory
	: method_factory<integrator_registry, integrator_product> {
};

}

#endif
