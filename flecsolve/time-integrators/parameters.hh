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
#ifndef FLECSI_LINALG_TIME_INTEGRATOR_PARAMETERS_H
#define FLECSI_LINALG_TIME_INTEGRATOR_PARAMETERS_H

#include <boost/program_options/options_description.hpp>
#include <limits>

#include "flecsolve/util/traits.hh"
#include "flecsolve/util/config.hh"
#include "flecsolve/operators/handle.hh"

namespace flecsolve::time_integrator {

namespace po = boost::program_options;

struct base_settings {
	double initial_time;
	double final_time;
	int max_steps;
	double max_dt;
	double min_dt;
	double initial_dt;
};

struct base_options : with_label {
	using settings_type = base_settings;
	explicit base_options(const char * pre) : with_label(pre) {}

	auto operator()(settings_type & s) {
		po::options_description desc;
		// clang-format off
		desc.add_options()
			(label("initial-time").c_str(), po::value<double>(&s.initial_time)->required(), "initial time for time integrator")
			(label("final-time").c_str(), po::value<double>(&s.final_time)->required(), "final time for time integrator")
			(label("max-steps").c_str(), po::value<int>(&s.max_steps)->required(), "maximum number of steps for time integrator")
			(label("max-dt").c_str(), po::value<double>(&s.max_dt)->default_value(std::numeric_limits<double>::max()), "maximum time step")
			(label("min-dt").c_str(), po::value<double>(&s.min_dt)->default_value(std::numeric_limits<double>::min()), "minimum time step")
			(label("initial-dt").c_str(), po::value<double>(&s.initial_dt)->default_value(0), "initial time step");
		// clang-format on
		return desc;
	}
};

template<class S, class O, class W>
struct parameters : S {
	template<class Work>
	parameters(const S & s, op::handle<O> op, Work && work)
		: S(s), op(op), work(std::forward<Work>(work)) {}

	auto & get_operator() {
		return op.get();
	}

	auto options() {}

	op::handle<O> op;
	std::decay_t<W> work;
};

}

#endif
