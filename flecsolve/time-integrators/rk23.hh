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
#ifndef FLECSI_LINALG_TIME_INTEGRATOR_RK23_H
#define FLECSI_LINALG_TIME_INTEGRATOR_RK23_H

#include <functional>
#include <type_traits>
#include <utility>

#include "flecsolve/solvers/solver_settings.hh"
#include "flecsolve/time-integrators/parameters.hh"
#include "flecsolve/time-integrators/base.hh"
#include "flecsolve/util/future.hh"

namespace flecsolve::time_integrator::rk23 {

struct settings : base_settings {
	float safety_factor;
	float atol;
	bool use_fixed_dt;
};

struct stepper_settings {
};

struct options : base_options {
	using settings_type = settings;
	explicit options(const char * pre) : base_options(pre) {}

	auto operator()(settings & s) {
		auto desc = base_options::operator()(s);

		// clang-format off
		desc.add_options()
			(label("safety-factor").c_str(), po::value<float>(&s.safety_factor)->default_value(0.9), "safety factor")
			(label("atol").c_str(), po::value<float>(&s.atol)->default_value(1e-9), "absolute tolerance")
			(label("use-fixed-dt").c_str(), po::value<bool>(&s.use_fixed_dt)->default_value(false), "use fixed dt");
		// clang-format on

		return desc;
	}
};
template<class Op, class Work, class Settings = settings>
struct parameters : time_integrator::parameters<Settings, Op, Work> {
	using base = time_integrator::parameters<Settings, Op, Work>;

	template<
		class W,
		class S = Settings,
		std::enable_if_t<!std::is_same_v<S, stepper_settings>, bool> = true>
	parameters(const S & s, op::handle<Op> op, W && work)
		: base(s, op, std::forward<W>(work)) {}

	template<class W,
	         class S = Settings,
	         std::enable_if_t<std::is_same_v<S, stepper_settings>, bool> = true>
	parameters(op::handle<Op> op, W && work)
		: base(stepper_settings{}, op, std::forward<W>(work)) {}

	auto & get_work() { return unwrap_work(this->work); }

private:
	template<class T>
	static T & unwrap_work(T & w) {
		return w;
	}

	template<class T>
	static T & unwrap_work(std::reference_wrapper<T> w) {
		return w.get();
	}
};
template<class O, class W>
parameters(const settings &, op::handle<O>, W &&) -> parameters<O, W>;
template<class O, class W>
parameters(op::handle<O>, W &&) -> parameters<O, W, stepper_settings>;

enum workvecs : std::size_t { k1, k2, k3, k4, z, next, nvecs };

static inline work_factory<workvecs::nvecs> make_work;
template<std::size_t Version = 0>
using topo_work = topo_work_base<workvecs::nvecs, Version>;

template<class O, class W>
struct stepper {
	using P = parameters<O, W, stepper_settings>;

	stepper(P p) : params(std::move(p)) {}

	template<class DeltaT,
	         class Curr,
	         class Out,
	         std::enable_if_t<is_scalar_or_future_v<std::decay_t<DeltaT>,
	                                                typename Curr::scalar>,
	                          bool> = true>
	void advance(DeltaT dt, Curr & curr, Out & out) {
		auto & F = params.get_operator();
		auto & [k1, k2, k3, k4, z, next] = params.get_work();

		auto dt_1_2 = 0.5 * dt;
		auto dt_3_4 = 0.75 * dt;
		auto dt_1_9 = (1. / 9.) * dt;
		auto dt_1_72 = (1. / 72.) * dt;

		// k1 = f(tn, un)
		F.apply(curr, k1);
		// u* = un + k1 * dt/2
		next.axpy(std::move(dt_1_2), k1, curr);
		// k2 = f(t+dt/2, u*)
		F.apply(next, k2);
		// u* = un + 0.75 *k2 * dt
		next.axpy(std::move(dt_3_4), k2, curr);
		// k3 = f(t + 0.75dt, u*)
		F.apply(next, k3);

		next.linear_sum(2.0, k1, 3.0, k2);
		next.axpy(4.0, k3, next);
		next.axpy(std::move(dt_1_9), next, curr);

		F.apply(next, k4);

		z.linear_sum(-5., k1, 6., k2);
		z.axpy(8., k3, z);
		z.axpy(-9., k4, z);
		z.scale(std::move(dt_1_72));
		out.copy(next);
	}

protected:
	P params;
};
template<class O, class W>
stepper(parameters<O, W, stepper_settings>) -> stepper<O, W>;

template<class O, class W>
struct integrator : base<parameters<O, W>> {
	using P = parameters<O, W>;
	using base<P>::params;
	using base<P>::current_dt;
	using base<P>::current_time;
	using base<P>::assert_can_advance;

	integrator(P p) : base<P>(std::move(p)), total_step_rejects(0) {}

	template<class Curr, class Out>
	void advance(double dt, Curr & curr, Out & out) {
		assert_can_advance();
		current_dt = dt;
		stepper(parameters{params.op, std::ref(params.work)})
			.advance(dt, curr, out);
	}

	bool check_solution() {
		auto & z = std::get<workvecs::z>(params.work);

		auto err_est = z.l2norm().get();

		if ((err_est < params.atol) ||
		    (std::fabs(current_dt - params.min_dt) < 1e-10))
			return true;

		return false;
	}

	double get_next_dt(bool good_solution) {
		double next_dt;
		if (params.use_fixed_dt) {
			next_dt = std::min(current_dt, params.final_time - current_time);
		}
		else {
			auto & z = std::get<workvecs::z>(params.work);
			auto est_err = z.l2norm().get();
			next_dt = params.safety_factor * current_dt *
			          std::pow(params.atol / est_err, 1. / 3.);
			next_dt = std::min(std::max(next_dt, params.min_dt), params.max_dt);
			next_dt = std::min(next_dt, params.final_time - current_time);
			if (not good_solution) {
				++total_step_rejects;
			}
		}
		return next_dt;
	}

protected:
	int total_step_rejects;
};
template<class O, class W>
integrator(parameters<O, W>) -> integrator<O, W>;

}

#endif
