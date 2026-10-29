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
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "flecsolve/operators/factory.hh"
#include "flecsolve/time-integrators/rk23.hh"
#include "flecsolve/time-integrators/rk45.hh"

namespace flecsolve::time_integrator {

enum class explicit_target { rk23, rk45 };

template<explicit_target T>
struct explicit_registry {
};

template<>
struct explicit_registry<explicit_target::rk23> {
	using settings = rk23::settings;
	using options = rk23::options;

	template<class V, class O>
	static auto make(const settings & s, const V & v, op::handle<O> rhs) {
		return rk23::integrator(rk23::parameters(s, rhs, rk23::make_work(v)));
	}

	template<class V, class O>
	static auto make_stepper(const V & v, op::handle<O> rhs) {
		return rk23::stepper(rk23::parameters(rhs, rk23::make_work(v)));
	}
};

template<>
struct explicit_registry<explicit_target::rk45> {
	using settings = rk45::settings;
	using options = rk45::options;

	template<class V, class O>
	static auto make(const settings & s, const V & v, op::handle<O> rhs) {
		return rk45::integrator(rk45::parameters(s, rhs, rk45::make_work(v)));
	}

	template<class V, class O>
	static auto make_stepper(const V & v, op::handle<O> rhs) {
		return rk45::stepper(rk45::parameters(rhs, rk45::make_work(v)));
	}
};

inline std::istream & operator>>(std::istream & in, explicit_target & target) {
	std::string token;
	in >> token;
	if (token == "rk23")
		target = explicit_target::rk23;
	else if (token == "rk45")
		target = explicit_target::rk45;
	else
		in.setstate(std::ios_base::failbit);
	return in;
}

struct explicit_factory_policy {
	using target = explicit_target;
	using targets = includes<target::rk23, target::rk45>;

	template<target T>
	using registry = explicit_registry<T>;
};

// Method-specific empty settings keep the configuration variant unambiguous:
// RK23 and RK45 share the same underlying stepper_settings type.
template<explicit_target T>
struct stepper_registry {
	using settings = null_settings<T>;
	using options = null_options<T>;

	template<class V, class O>
	static auto make(const settings &, const V & v, op::handle<O> rhs) {
		return explicit_registry<T>::make_stepper(v, std::move(rhs));
	}
};

struct stepper_factory_policy {
	using target = explicit_target;
	using targets = includes<target::rk23, target::rk45>;

	template<target T>
	using registry = stepper_registry<T>;
};

// The application supplies timesteps and manages time and solution storage.
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
			[&](auto & stepper) { stepper.advance(std::move(dt), curr, out); },
			var);
	}

private:
	Variant var;
};

struct stepper_factory {
	using implementation = op::factory<stepper_factory_policy>;
	using target = implementation::target;
	using settings = implementation::settings;
	using options = implementation::options;

	template<class V, class O>
	static auto make(const settings & s, const V & v, op::handle<O> rhs) {
		auto var = implementation::make_policy(s, v, std::move(rhs));
		return stepper_product<decltype(var)>(std::move(var));
	}

	template<class V, class O>
	static auto make(target method, const V & v, op::handle<O> rhs) {
		return make(settings{method, implementation::make_settings(method)},
		            v,
		            std::move(rhs));
	}
};

// advance() computes a candidate solution; update() commits an accepted step.
// The caller swaps solution vectors on acceptance and retries rejected steps.
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

struct explicit_factory {
	using impl = op::factory<explicit_factory_policy>;
	using settings = impl::settings;
	using options = impl::options;

	// rhs must apply the explicit right-hand side, rather than an implicit
	// system such as I - gamma * F. Its handle preserves caller ownership.
	template<class V, class O>
	static auto make(const settings & s, const V & v, op::handle<O> rhs) {
		auto var = impl::make_policy(s, v, std::move(rhs));
		return integrator_product<decltype(var)>(std::move(var));
	}
};

}

#endif
