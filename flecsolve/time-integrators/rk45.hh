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
#ifndef FLECSI_LINALG_TIME_INTEGRATOR_RK45_H
#define FLECSI_LINALG_TIME_INTEGRATOR_RK45_H

#include <functional>
#include <type_traits>
#include <utility>

#include "flecsolve/solvers/solver_settings.hh"
#include "flecsolve/time-integrators/parameters.hh"
#include "flecsolve/time-integrators/rk23.hh"
#include "flecsolve/time-integrators/base.hh"
#include "flecsolve/util/future.hh"

namespace flecsolve::time_integrator::rk45 {

struct settings : rk23::settings {
};
struct options : rk23::options {
	using settings_type = settings;
	explicit options(const char * pre) : rk23::options(pre) {}
};
using stepper_settings = rk23::stepper_settings;

template<class Op, class Work, class Settings = settings>
struct parameters : rk23::parameters<Op, Work, Settings> {
	using base = rk23::parameters<Op, Work, Settings>;
	using base::base;
};
template<class O, class W>
parameters(const settings &, op::handle<O>, W &&) -> parameters<O, W>;
template<class O, class W>
parameters(op::handle<O>, W &&) -> parameters<O, W, stepper_settings>;

enum workvecs : std::size_t { k1, k2, k3, k4, k5, k6, z, next, nvecs };

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
	         std::enable_if_t<
				 is_scalar_or_future_v<std::decay_t<DeltaT>, typename Curr::scalar>,
				 bool> = true>
	void advance(DeltaT dt, Curr & curr, Out & out) {
		auto & F = params.get_operator();
		auto & [k1, k2, k3, k4, k5, k6, z, next] = params.get_work();

		auto dt_1_4 = 0.25 * dt;
		auto dt_3_32 = (3. / 32.) * dt;
		auto dt_9_32 = (9. / 32.) * dt;
		auto dt_1932_2197 = (1932. / 2197.) * dt;
		auto dt_neg_7200_2197 = (-7200. / 2197.) * dt;
		auto dt_7296_2197 = (7296. / 2197.) * dt;
		auto dt_439_216 = (439. / 216.) * dt;
		auto dt_neg_8 = -8. * dt;
		auto dt_3680_513 = (3680. / 513.) * dt;
		auto dt_neg_845_4104 = (-845. / 4104.) * dt;
		auto dt_neg_8_27 = (-8. / 27.) * dt;
		auto dt_2 = 2. * dt;
		auto dt_neg_3544_2565 = (-3544. / 2565.) * dt;
		auto dt_1859_4104 = (1859. / 4104.) * dt;
		auto dt_neg_11_40 = (-11. / 40.) * dt;
		auto dt_25_216 = (25. / 216.) * dt;
		auto dt_1408_2565 = (1408. / 2565.) * dt;
		auto dt_2197_4104 = (2197. / 4104.) * dt;
		auto dt_neg_1_5 = -0.2 * dt;
		auto dt_16_135 = (16. / 135.) * dt;
		auto dt_6656_12825 = (6656. / 12825.) * dt;
		auto dt_28561_56430 = (28561. / 56430.) * dt;
		auto dt_neg_9_50 = (-9. / 50.) * dt;
		auto dt_2_55 = (2. / 55.) * dt;

		F.apply(curr, k1);
		next.axpy(std::move(dt_1_4), k1, curr);
		F.apply(next, k2);

		next.axpy(std::move(dt_3_32), k1, curr);
		next.axpy(std::move(dt_9_32), k2, next);

		F.apply(next, k3);

		next.axpy(std::move(dt_1932_2197), k1, curr);
		next.axpy(std::move(dt_neg_7200_2197), k2, next);
		next.axpy(std::move(dt_7296_2197), k3, next);

		F.apply(next, k4);

		next.axpy(std::move(dt_439_216), k1, curr);
		next.axpy(std::move(dt_neg_8), k2, next);
		next.axpy(std::move(dt_3680_513), k3, next);
		next.axpy(std::move(dt_neg_845_4104), k4, next);

		F.apply(next, k5);

		next.axpy(std::move(dt_neg_8_27), k1, curr);
		next.axpy(std::move(dt_2), k2, next);
		next.axpy(std::move(dt_neg_3544_2565), k3, next);
		next.axpy(std::move(dt_1859_4104), k4, next);
		next.axpy(std::move(dt_neg_11_40), k5, next);

		F.apply(next, k6);

		z.axpy(std::move(dt_25_216), k1, curr);
		z.axpy(std::move(dt_1408_2565), k3, z);
		z.axpy(std::move(dt_2197_4104), k4, z);
		z.axpy(std::move(dt_neg_1_5), k5, z);

		next.axpy(std::move(dt_16_135), k1, curr);
		next.axpy(std::move(dt_6656_12825), k3, next);
		next.axpy(std::move(dt_28561_56430), k4, next);
		next.axpy(std::move(dt_neg_9_50), k5, next);
		next.axpy(std::move(dt_2_55), k6, next);

		z.subtract(next, z);
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
