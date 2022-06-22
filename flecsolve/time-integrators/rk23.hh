#ifndef FLECSI_LINALG_TIME_INTEGRATOR_RK23_H
#define FLECSI_LINALG_TIME_INTEGRATOR_RK23_H

#include "flecsolve/solvers/solver_settings.hh"
#include "flecsolve/time-integrators/parameters.hh"

namespace flecsolve::time_integrator::rk23 {

template<class Op, class Work>
struct parameters : time_integrator::parameters<Op, Work> {
	using base = time_integrator::parameters<Op, Work>;
	using base::desc;
	using base::label;

	template<class O, class W>
	parameters(const char * pre, O && op, W && work)
		: base(pre, std::forward<O>(op), std::forward<W>(work)) {
		// clang-format off
		desc.add_options()
			(label("safety-factor").c_str(), po::value<float>(&safety_factor)->default_value(0.9), "safety factor")
			(label("atol").c_str(), po::value<float>(&atol)->default_value(1e-9), "absolute tolerance")
			(label("use-fixed-dt").c_str(), po::value<bool>(&use_fixed_dt)->default_value(false), "use fixed dt");
		// clang-format on
	}

	float safety_factor;
	float atol;
	bool use_fixed_dt;
};
template<class O, class W>
parameters(const char *, O &&, W &&)
	-> parameters<std::decay_t<O>, std::decay_t<W>>;

enum workvecs : std::size_t { k1, k2, k3, k4, z, nvecs };

template<std::size_t Version = 0>
using topo_work = topo_work_base<workvecs::nvecs, Version>;

template<class P>
struct integrator {

	integrator(P p)
		: params(std::move(p)), current_time(params.initial_time),
		  current_dt(params.initial_dt), old_dt(params.initial_dt),
		  total_step_rejects(0) {}

	template<class Curr, class Next>
	void advance(double dt, Curr & curr, Next & next) {
		current_dt = dt;
		auto & F = params.get_operator();
		auto & [k1, k2, k3, k4, z] = params.work;

		// k1 = f(tn, un)
		F.apply(curr, k1);
		// u* = un + k1 * dt/2
		next.axpy(0.5 * dt, k1, curr);
		// k2 = f(t+dt/2, u*)
		F.apply(next, k2);
		// u* = un + 0.75 *k2 * dt
		next.axpy(0.5 * dt, k2, curr);
		// k3 = f(t + 0.75dt, u*)
		F.apply(next, k3);

		next.linear_sum(2.0, k1, 3.0, k2);
		next.axpy(4.0, k3, next);
		next.axpy(dt / 9.0, next, curr);

		F.apply(next, k4);

		z.linear_sum(-5., k1, 6., k2);
		z.axpy(8., k3, z);
		z.axpy(-9., k4, z);
		z.scale(dt / 72., z);
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
			auto est_err = z->l2norm().get();
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
	P params;
	double current_time;
	double current_dt;
	double old_dt;
	int total_step_rejects;
};
template<class P>
integrator(P) -> integrator<P>;

}

#endif
