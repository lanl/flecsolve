#ifndef FLECSI_LINALG_TIME_INTEGRATOR_RK45_H
#define FLECSI_LINALG_TIME_INTEGRATOR_RK45_H

#include "flecsolve/solvers/solver_settings.hh"
#include "flecsolve/time-integrators/parameters.hh"
#include "flecsolve/time-integrators/rk23.hh"
#include "flecsolve/time-integrators/base.hh"

namespace flecsolve::time_integrator::rk45 {

template<class Op, class Work>
struct parameters : rk23::parameters<Op, Work> {
	template<class O, class W>
	parameters(const char * pre, O && op, W && work)
		: rk23::parameters<Op, Work>(pre,
	                                 std::forward<O>(op),
	                                 std::forward<W>(work)) {}
};
template<class O, class W>
parameters(const char *, O &&, W &&)
	-> parameters<std::decay_t<O>, std::decay_t<W>>;

enum workvecs : std::size_t { k1, k2, k3, k4, k5, k6, z, next, nvecs };

template<std::size_t Version = 0>
using topo_work = topo_work_base<workvecs::nvecs, Version>;

template<class O, class W>
struct integrator : base<parameters<O, W>> {
	using P = parameters<O, W>;
	using base<P>::params;
	using base<P>::current_dt;
	using base<P>::current_time;

	integrator(P p) : base<P>(std::move(p)), total_step_rejects(0) {}

	template<class Curr, class Out>
	void advance(double dt, Curr & curr, Out & out) {
		current_dt = dt;
		auto & F = params.get_operator();
		auto & [k1, k2, k3, k4, k5, k6, z, next] = params.work;

		F.apply(curr, k1);
		next.axpy(0.25 * dt, k1, curr);
		F.apply(next, k2);

		next.axpy(3.0 * dt / 32., k1, curr);
		next.axpy(9.0 * dt / 32., k2, next);

		F.apply(next, k3);

		next.axpy(1932. * dt / 2197., k1, curr);
		next.axpy(-7200. * dt / 2197., k2, next);
		next.axpy(7296. * dt / 2197., k3, next);

		F.apply(next, k4);

		next.axpy(439. * dt / 216., k1, curr);
		next.axpy(-8. * dt, k2, next);
		next.axpy(3680. * dt / 513., k3, next);
		next.axpy(-845. * dt / 4104., k4, next);

		F.apply(next, k5);

		next.axpy(-8. * dt / 27., k1, curr);
		next.axpy(2. * dt, k2, next);
		next.axpy(-3544. * dt / 2565., k3, next);
		next.axpy(1859. * dt / 4104., k4, next);
		next.axpy(-11. * dt / 40., k5, next);

		F.apply(next, k6);

		z.axpy(25. * dt / 216., k1, curr);
		z.axpy(1408. * dt / 2565., k3, z);
		z.axpy(2197. * dt / 4104., k4, z);
		z.axpy(-0.2 * dt, k5, z);

		next.axpy(16. * dt / 135., k1, curr);
		next.axpy(6656. * dt / 12825., k3, next);
		next.axpy(28561. * dt / 56430., k4, next);
		next.axpy(-9. * dt / 50., k5, next);
		next.axpy(2. * dt / 55., k6, next);

		z.subtract(next, z);
		out.copy(next);
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
			if (good_solution) {
				auto & z = std::get<workvecs::z>(params.work);
				auto err_est = z.l2norm().get();
				next_dt = params.safety_factor * current_dt *
				          std::pow((params.atol / err_est), 1. / 5.);
				next_dt =
					std::min(std::max(next_dt, params.min_dt), params.max_dt);
				next_dt = std::min(next_dt, params.final_time - current_time);
			}
			else {
				next_dt = params.safety_factor * current_dt;
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
