#ifndef FLECSI_LINALG_TIME_INTEGRATOR_PARAMETERS_H
#define FLECSI_LINALG_TIME_INTEGRATOR_PARAMETERS_H

#include <boost/program_options/options_description.hpp>
#include <limits>

#include "flecsolve/util/traits.hh"

namespace flecsolve::time_integrator {

namespace po = boost::program_options;
template<class O, class W>
struct parameters {
	template<class Op, class Work>
	parameters(const char * pre, Op && op, Work && work)
		: op(std::forward<Op>(op)), work(std::forward<Work>(work)),
		  prefix(pre) {
		// clang-format off
		desc.add_options()
			(label("initial-time").c_str(), po::value<double>(&initial_time)->required(), "initial time for time integrator")
			(label("final-time").c_str(), po::value<double>(&final_time)->required(), "final time for time integrator")
			(label("max-steps").c_str(), po::value<int>(&max_steps)->required(), "maximum number of steps for time integrator")
			(label("max-dt").c_str(), po::value<double>(&max_dt)->default_value(std::numeric_limits<double>::max()), "maximum time step")
			(label("min-dt").c_str(), po::value<double>(&min_dt)->default_value(std::numeric_limits<double>::min()), "minimum time step")
			(label("initial-dt").c_str(), po::value<double>(&initial_dt)->default_value(0), "initial time step");
		// clang-format on
	}

	auto & get_operator() {
		if constexpr (is_reference_wrapper_v<O>)
			return op.get();
		else
			return op;
	}

	auto & options() { return desc; }
	const auto & options() const { return desc; }

	po::options_description desc;
	double initial_time;
	double final_time;
	int max_steps;
	double max_dt;
	double min_dt;
	double initial_dt;
	O op;
	W work;

protected:
	std::string prefix;

	std::string label(const char * suf) { return {prefix + "." + suf}; }
};

}

#endif
