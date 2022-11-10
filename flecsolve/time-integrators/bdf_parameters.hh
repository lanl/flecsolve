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

namespace po = boost::program_options;

struct parameters_gen : with_label {

	parameters_gen(const char * pre) : with_label(pre) {}

	auto options() {
		po::options_description desc;
		// clang-format off
		desc.add_options()
			(label("use-predictor").c_str(), po::value<bool>(&use_predictor)->default_value(true), "use a predictor")
			(label("has-source-term").c_str(), po::value<bool>(&has_source_term)->default_value(false), "has source term")
			(label("use-initial-predictor").c_str(), po::value<bool>(&use_initial_predictor)->default_value(true), "use an initial predictor")
			(label("predictor").c_str(), po::value<bdf::predictor>(&predictor)->required(), "Predictor type (AB2 or Leapfrog)")
			(label("timestep-selection-strategy").c_str(), po::value<strategy>(&timestep_strategy)->required(),
			 "Timestep selection strategy (truncation error or constant or final constant or limit relative change)")
			// these bounds are based on the paper by Emmrich, 2008 for nonlinear evolution
			(label("dt-cut-lower-bound").c_str(), po::value<double>(&dt_cut_lower_bound)->default_value(0.58754407), "dt-cut-lower-bound")
			(label("dt-growth-upper-bound").c_str(), po::value<double>(&dt_growth_upper_bound)->default_value(1.702), "dt-growth-upper-bound")
			(label("number-of-time-intervals").c_str(), po::value<int>(&number_of_time_intervals)->default_value(100), "Number of time intervals (final constant strategy)")
			(label("integrator").c_str(), po::value<bdf::method>(&integrator)->required(), "Implicit integrator (BE or CN or BDF{2-6}")
			(label("starting-integrator").c_str(), po::value<bdf::method>(&starting_integrator)->required()->notifier([](const bdf::method & m) {
				flog_assert(memory_size(m) == 1, "BE or CN must be used for starting integrator");}), "one step integrator (BE or CN)")
			(label("calculate-time-trunc-error").c_str(), po::value<bool>(&calculate_time_trunc_error), "Whether to calculate the time truncation error")
			(label("time-trunc-error-norm").c_str(), po::value<vec::norm_type>(&time_trunc_err_norm), "Norm time used for time truncation error")
			(label("target-relative-change").c_str(), po::value<double>(&target_relative_change), "Relative change to target in time scale")
			(label("use-pi-controller").c_str(), po::value<bool>(&use_pi_controller)->default_value(true), "Whether to use the PI controller")
			(label("pi-controller-type").c_str(), po::value<bdf::controller>(&pi_controller_type)->default_value(bdf::controller::pc4_7, "PC.4.7"), "Type of PI controller")
			(label("control-timestep-variation").c_str(), po::value<bool>(&control_timestep_variation)->default_value(false), "Control timestep variation")
			(label("time-error-scaling").c_str(), po::value<bdf::error_scaling>(&time_error_scaling)->default_value(error_scaling::fixed_scaling, "fixed-scaling"), "Time error scaling")
			(label("trunc-error-rtol").c_str(), po::value<double>(&time_rtol)->default_value(1e-9), "Relative tolerance for truncation error")
			(label("trunc-error-atol").c_str(), po::value<double>(&time_atol)->default_value(1e-15), "Absolute tolerance for truncation error")
			(label("combine-timestep-estimators").c_str(), po::value<bool>(&combine_timestep_estimators)->default_value(false), "Combine timestep estimators")
			(label("problem-scales").c_str(), po::value<std::vector<double>>(&problem_scales)->multitoken(), "Fixed scaling for problem");
		// clang-format on

		return desc;
	}

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
};

template<class Op, class Work, class Solver, bool use_factory>
struct parameters {};

template<class Op, class Work, class Solver>
struct parameters<Op, Work, Solver, false>
	: parameters_gen, time_integrator::parameters<Op, Work> {
	using base = time_integrator::parameters<Op, Work>;
	using base::label;

	template<class O, class W, class S>
	parameters(const char * pre, O && op, W && work, S && solver)
		: parameters_gen(pre),
		  base(pre, std::forward<O>(op), std::forward<W>(work)),
		  solver(std::forward<S>(solver)) {}

	auto & get_solver() {
		if constexpr (is_reference_wrapper_v<Solver>)
			return solver.get();
		else
			return solver;
	}

	auto options() {
		auto desc = base::options();
		auto desc_gen = parameters_gen::options();
		desc.add(desc_gen);

		return desc;
	}

protected:
	std::decay_t<Solver> solver;
};

template<class Op, class Work, class Factory>
struct parameters<Op, Work, Factory, true>
	: parameters_gen, time_integrator::parameters<Op, Work> {
	using base = time_integrator::parameters<Op, Work>;
	using base::get_operator;
	using base::label;
	using base::work;

	using factory_t = std::decay_t<Factory>;

	template<class O, class W, class F>
	parameters(const char * pre, O && op, W && work, F && f)
		: parameters_gen(pre),
		  base(pre, std::forward<O>(op), std::forward<W>(work)),
		  factory(std::forward<F>(f)) {}

	auto get_solver() {
		if (!factory.has_solver())
			factory.create(std::get<0>(work), std::ref(get_operator()));
		return op::shell([=](auto & x, auto & y) {
			return factory.solve(
				x, y, std::get<0>(work), std::ref(get_operator()));
		});
	}

	auto options() {
		auto desc = base::options();
		desc.add_options()(label("solver").c_str(),
		                   po::value<std::string>()->required()->notifier(
							   [this](const std::string & name) {
								   factory.set_options_name(name);
							   }),
		                   "solver name");

		desc.add(factory.options(
			std::bind(std::mem_fun(&parameters::create_solver_parameters),
		              this,
		              std::placeholders::_1)));

		auto desc_gen = parameters_gen::options();
		desc.add(desc_gen);

		return desc;
	}

protected:
	void create_solver_parameters(const typename factory_t::registry & reg) {
		factory.create_parameters(
			reg, std::get<0>(work), std::ref(get_operator()));
	}
	factory_t factory;
};

template<class O, class W, class S>
parameters(const char *, O &&, W &&, S &&)
	-> parameters<O, W, S, !op::is_solver_v<S, std::tuple_element_t<0, W>>>;
}
#endif
