#ifndef FLECSOLVE_TIME_INTEGRATOR_BDF_PARAMETERS_H
#define FLECSOLVE_TIME_INTEGRATOR_BDF_PARAMETERS_H

#include <functional>
#include <type_traits>

#include "flecsolve/util/config.hh"
#include "flecsolve/vectors/util.hh"
#include "flecsolve/solvers/solver_settings.hh"
#include "flecsolve/time-integrators/parameters.hh"
#include "flecsolve/solvers/factory.hh"

namespace flecsolve::time_integrator::bdf {

enum class method { cn, be, bdf2, bdf3, bdf4, bdf5, bdf6 };
enum class predictor { ab2, leapfrog };
enum class strategy {
	truncation_error,
	constant,
	final_constant,
	limit_relative_change
};
enum class controller { H211b, pc4_7, pc11, deadbeat };
enum class error_scaling { fixed_resolution, fixed_scaling };

int memory_size(method);

std::istream & operator>>(std::istream &, predictor &);
std::istream & operator>>(std::istream &, strategy &);
std::istream & operator>>(std::istream &, method &);
std::istream & operator>>(std::istream &, controller &);
std::istream & operator>>(std::istream &, error_scaling &);

struct settings : base_settings {
	bool use_predictor;
	bool use_initial_predictor;
	bool has_source_term;
	bool combine_timestep_estimators;
	bdf::predictor predictor;
	bdf::strategy timestep_strategy;
	double dt_cut_lower_bound;
	double dt_growth_upper_bound;
	int number_of_time_intervals;
	bdf::method integrator;
	bdf::method starting_integrator;
	bool calculate_time_trunc_error;
	vec::norm_type time_trunc_err_norm;
	double target_relative_change;
	bool use_pi_controller;
	bdf::controller pi_controller_type;
	bool control_timestep_variation;
	bdf::error_scaling time_error_scaling;
	double time_rtol, time_atol;
	std::vector<double> problem_scales;

	void validate() {
		flog_assert(memory_size(starting_integrator) == 1,
		            "Starting integrator must be CN or BE");

		if (use_predictor) {
			// override if using predictor
			calculate_time_trunc_error = true;
		}

		if (timestep_strategy == strategy::truncation_error) {
			// override if we are using the truncation error strategy
			calculate_time_trunc_error = true;
			use_predictor = true;
		}
		else {
			combine_timestep_estimators = false;
			control_timestep_variation = false;
			use_pi_controller = false;
		}

		if (calculate_time_trunc_error) {
			use_predictor = true;
			if (time_error_scaling == error_scaling::fixed_scaling) {
				flog_assert(problem_scales.size() > 0,
				            "Problem scales must be specified if using fixed "
				            "time error scaling");
			}
		}

		if (use_predictor) {
			if (memory_size(integrator) > 1) {
				flog_assert(
					predictor == bdf::predictor::leapfrog,
					"Only the leapfrog predictor is supported for BDF2-6");
			}
			else if (integrator == method::cn) {
				flog_assert(predictor == bdf::predictor::ab2,
				            "Valid option for Crank-Nicolson predictor is only "
				            "ab2 currently");
			}
		}
	}
};

struct options : base_options {
	using settings_type = settings;
	explicit options(const char * pre) : base_options(pre) {}

	auto operator()(settings & s) {
		auto desc = base_options::operator()(s);
		// clang-format off
		desc.add_options()
			(label("use-predictor").c_str(), po::value<bool>(&s.use_predictor)->default_value(true), "use a predictor")
			(label("has-source-term").c_str(), po::value<bool>(&s.has_source_term)->default_value(false), "has source term")
			(label("use-initial-predictor").c_str(), po::value<bool>(&s.use_initial_predictor)->default_value(true), "use an initial predictor")
			(label("predictor").c_str(), po::value<bdf::predictor>(&s.predictor)->required(), "Predictor type (AB2 or Leapfrog)")
			(label("timestep-selection-strategy").c_str(), po::value<strategy>(&s.timestep_strategy)->required(),
			 "Timestep selection strategy (truncation error or constant or final constant or limit relative change)")
			// these bounds are based on the paper by Emmrich, 2008 for nonlinear evolution
			(label("dt-cut-lower-bound").c_str(), po::value<double>(&s.dt_cut_lower_bound)->default_value(0.58754407), "dt-cut-lower-bound")
			(label("dt-growth-upper-bound").c_str(), po::value<double>(&s.dt_growth_upper_bound)->default_value(1.702), "dt-growth-upper-bound")
			(label("number-of-time-intervals").c_str(), po::value<int>(&s.number_of_time_intervals)->default_value(100), "Number of time intervals (final constant strategy)")
			(label("integrator").c_str(), po::value<bdf::method>(&s.integrator)->required(), "Implicit integrator (BE or CN or BDF{2-6}")
			(label("starting-integrator").c_str(), po::value<bdf::method>(&s.starting_integrator)->required()->notifier([](const bdf::method & m) {
				flog_assert(memory_size(m) == 1, "BE or CN must be used for starting integrator");}), "one step integrator (BE or CN)")
			(label("calculate-time-trunc-error").c_str(), po::value<bool>(&s.calculate_time_trunc_error), "Whether to calculate the time truncation error")
			(label("time-trunc-error-norm").c_str(), po::value<vec::norm_type>(&s.time_trunc_err_norm), "Norm time used for time truncation error")
			(label("target-relative-change").c_str(), po::value<double>(&s.target_relative_change), "Relative change to target in time scale")
			(label("use-pi-controller").c_str(), po::value<bool>(&s.use_pi_controller)->default_value(true), "Whether to use the PI controller")
			(label("pi-controller-type").c_str(), po::value<bdf::controller>(&s.pi_controller_type)->default_value(bdf::controller::pc4_7, "PC.4.7"), "Type of PI controller")
			(label("control-timestep-variation").c_str(), po::value<bool>(&s.control_timestep_variation)->default_value(false), "Control timestep variation")
			(label("time-error-scaling").c_str(), po::value<bdf::error_scaling>(&s.time_error_scaling)->default_value(error_scaling::fixed_scaling, "fixed-scaling"), "Time error scaling")
			(label("trunc-error-rtol").c_str(), po::value<double>(&s.time_rtol)->default_value(1e-9), "Relative tolerance for truncation error")
			(label("trunc-error-atol").c_str(), po::value<double>(&s.time_atol)->default_value(1e-15), "Absolute tolerance for truncation error")
			(label("combine-timestep-estimators").c_str(), po::value<bool>(&s.combine_timestep_estimators)->default_value(false), "Combine timestep estimators")
			(label("problem-scales").c_str(), po::value<std::vector<double>>(&s.problem_scales)->multitoken(), "Fixed scaling for problem");
		// clang-format on

		return desc;
	}
};

template<class Op, class Work, class Solver>
struct parameters : time_integrator::parameters<settings, Op, Work> {
	using base = time_integrator::parameters<settings, Op, Work>;

	template<class O, class W, class S>
	parameters(const settings & s, O && op, W && work, S && solver)
		: base(s, std::forward<O>(op), std::forward<W>(work)),
		  solver(std::forward<S>(solver)) {}

	auto & get_solver() {
		if constexpr (is_reference_wrapper_v<Solver>)
			return solver.get();
		else
			return solver;
	}

protected:
	std::decay_t<Solver> solver;
};

template<class O, class W, class S>
parameters(const settings &, O &&, W &&, S &&) -> parameters<O, W, S>;
}
#endif
