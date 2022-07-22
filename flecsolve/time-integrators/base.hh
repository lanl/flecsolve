#ifndef FLECSI_LINALG_TIME_INTEGRATOR_BASE_H
#define FLECSI_LINALG_TIME_INTEGRATOR_BASE_H

#include <utility>

namespace flecsolve::time_integrator {

template<class P>
struct base {
	base(P && p)
		: params(std::move(p)), current_time(params.initial_time),
		  current_dt(params.initial_dt), old_dt(params.initial_dt),
		  integrator_step(0) {}

	double get_current_time() const { return current_time; }

	double get_final_time() const { return params.final_time; }

	void update() {
		current_time += current_dt;
		++integrator_step;
	}

	int get_current_step() const { return integrator_step; }

	double get_current_dt() const { return current_dt; }

	bool fixed_dt() const { return params.use_fixed_dt; }

protected:
	P params;
	double current_time;
	double current_dt;
	double old_dt;
	int integrator_step;
};

template<class P>
struct implicit : base<P> {
	using base<P>::params;

	implicit(P && p) : base<P>(std::move(p)) {}

	auto & get_solver() { return params.solver; }

	const auto & get_solver() const { return params.solver; }
};

}

#endif
