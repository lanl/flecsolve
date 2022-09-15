#ifndef FLECSI_LINALG_TIME_INTEGRATOR_BDF_H
#define FLECSI_LINALG_TIME_INTEGRATOR_BDF_H

#include <limits>

#include "flecsi/flog.hh"
#include "flecsi/util/array_ref.hh"

#include "flecsolve/vectors/util.hh"
#include "flecsolve/solvers/solver_settings.hh"
#include "flecsolve/time-integrators/parameters.hh"
#include "flecsolve/time-integrators/base.hh"

namespace flecsolve::time_integrator::bdf {

enum workvec : std::size_t {
	rhs,
	source,
	current_sol,
	scratch,
	previous_function,
	current_function,
	scratch_function,
	old_td,
	predict,
	time_deriv,
	nwork
};

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

inline constexpr std::size_t work_size = workvec::nwork + 6; // max for bdf6

template<std::size_t Version = 0>
using topo_work = topo_work_base<work_size, Version>;

int memory_size(method);
short order(method meth);

std::istream & operator>>(std::istream &, predictor &);
std::istream & operator>>(std::istream &, strategy &);
std::istream & operator>>(std::istream &, method &);
std::istream & operator>>(std::istream &, controller &);
std::istream & operator>>(std::istream &, error_scaling &);

template<class Op, class Work, class Solver>
struct parameters : time_integrator::parameters<Op, Work> {
	using base = time_integrator::parameters<Op, Work>;
	using base::label;

	template<class O, class W, class S>
	parameters(const char * pre, O && op, W && work, S && solver)
		: base(pre, std::forward<O>(op), std::forward<W>(work)),
		  solver(std::forward<S>(solver)) {}

	auto & get_solver() {
		if constexpr (is_reference_wrapper_v<Solver>)
			return solver.get();
		else
			return solver;
	}

	auto options() {
		auto desc = base::options();
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

protected:
	std::decay_t<Solver> solver;
};
template<class O, class W, class S>
parameters(const char *, O &&, W &&, S &&) -> parameters<O, W, S>;

template<class O, class W, class S>
struct integrator : base<parameters<O, W, S>> {
	using P = parameters<O, W, S>;
	using base<P>::params;
	using base<P>::current_dt;
	using base<P>::old_dt;
	using base<P>::current_time;
	using base<P>::steps_remaining;
	using base<P>::integrator_step;

	integrator(P p)
		: base<P>(std::move(p)),
		  prev(memory_size(params.integrator), std::ref(params.work)),
		  solver_success{false}, total_steprejects{0} {
		params.validate();

		auto & scratch = getvec<workvec::scratch>();
		scratch.zero();
	}

	template<class Curr, class Out>
	void advance(double dt, bool first_step, Curr & curr, Out & out) {
		flog_assert(curr != out,
		            "BDF integrator: curr cannot be the same as out");
		flog_assert(steps_remaining() && (current_time < params.final_time),
		            "BDF integrator: already finished integrating");

		prev[0].solution.copy(curr);
		current_dt = dt;

		set_initial_guess(first_step, current_time, current_dt, old_dt);

		auto & solver = params.get_solver();
		auto & rhs = getvec<workvec::rhs>();
		auto & source = getvec<workvec::source>();
		auto & sol = getvec<workvec::current_sol>();

		rhs.scale(-1., source);
		auto info = solver.apply(rhs, sol);
		solver_success = info.success();
		out.copy(sol);
	}

	double get_time_operator_scaling() const { return gamma; }

	bool check_solution() {
		bool check_passed{false};

		if (solver_success) {
			if (params.calculate_time_trunc_error)
				calculate_temporal_truncation_error();

			if (params.timestep_strategy == strategy::truncation_error) {
				check_passed = (time_trunc_err_est <= 1.0);

				if (check_passed) {
					prev_successive_rejects = (current_steprejects > 1);
					current_steprejects = 0;
				}
				else {
					++current_steprejects;
				}
			}
			else
				check_passed = true;
		}
		else {
			++current_steprejects;
		}

		if (!check_passed)
			++total_steprejects;

		return check_passed;
	}

	void update() {
		current_time += current_dt;
		this->new_time = current_time;
		if (params.use_predictor) {
			if (params.predictor == predictor::ab2) {
				auto & old_td = getvec<workvec::old_td>();
				auto & curr_fn = getvec<workvec::current_function>();
				std::swap(old_td, curr_fn);
			}
			estimate_time_derivative();
		}

		auto & curr_sol = getvec<workvec::current_sol>();
		prev.push(curr_sol, current_dt);

		++integrator_step;
	}

	double get_next_dt(bool good_solution) {
		double tmp_dt = current_dt;
		if (params.timestep_strategy == strategy::truncation_error) {
			if (good_solution) {
				current_dt = estimate_dt_with_truncation_error_estimates(
					current_dt, good_solution);

				if (params.max_dt < current_dt)
					current_dt = params.max_dt;
			}
			else {
				if (solver_success) {
					current_dt = estimate_dt_with_truncation_error_estimates(
						current_dt, good_solution);
				}
				else {
					current_dt = params.dt_cut_lower_bound * current_dt;
				}
			}
			evaluate_predictor();
			auto & predict = getvec<workvec::predict>();
			auto & op = params.get_operator();
			bool valid_vector = op.is_valid(predict);
			if (!valid_vector) {
				int number_of_predictor_precheck_events = 10;
				for (int i = 0; i < number_of_predictor_precheck_events; ++i) {
					current_dt = 0.5 * current_dt;
					evaluate_predictor();
					valid_vector = op.is_valid(predict);
				}
			}
		}
		else {
			if (good_solution) {
				if (params.timestep_strategy == strategy::constant)
					current_dt = params.initial_dt;
				else if (params.timestep_strategy == strategy::final_constant) {
					static int i = 1;
					if (i < params.number_of_time_intervals) {
						current_dt =
							params.initial_dt +
							((double)i) /
								((double)params.number_of_time_intervals) *
								(params.max_dt - params.initial_dt);
						++i;
					}
					else {
						current_dt = params.max_dt;
					}
				}
				else if (params.timestep_strategy ==
				         strategy::limit_relative_change) {
					current_dt = estimate_dynamical_time_scale(current_dt);
				}
			}
			else
				current_dt = 0.;
		}

		// now set old dt once it has been used
		if (good_solution)
			old_dt = tmp_dt;

		current_dt = std::min(std::min(current_dt, params.max_dt),
		                      params.final_time - current_time);
		return current_dt;
	}

	int num_step_rejects() const { return total_steprejects; }

protected:
	method get_current_integrator() const {
		if (integrator_step == 0) {
			return params.starting_integrator;
		}
		else if ((integrator_step + 1) < memory_size(params.integrator)) {
			return static_cast<method>(integrator_step + 1);
		}
		else {
			return params.integrator;
		}
	}

	void set_initial_guess(bool first_step,
	                       double current_time,
	                       double current_dt,
	                       double old_dt) {
		(void)current_time;
		(void)current_dt;
		(void)old_dt;

		this->first_step = first_step;
		auto current_integrator = get_current_integrator();
		// compute f(u_{n+1}) and store it for the timestep
		if (current_integrator == method::cn) {
			auto & scratch = getvec<workvec::scratch>();
			auto & prev_function = getvec<workvec::previous_function>();
			auto & time_op = params.get_operator();
			scratch.copy(prev[0].solution);
			time_op.apply_rhs(scratch, prev_function);
		}

		if (params.use_predictor) {
			auto & predict = getvec<workvec::predict>();
			if (first_step && (!params.use_initial_predictor)) {
				// for first step use constant extrapolation in time
				predict.copy(prev[0].solution);
			}
			else
				evaluate_predictor();

			// we call reInitializeVector because it postprocesses a vector
			// to ensure it is a valid vector for E & T values. The routine
			// itself should be renamed as clearly there are multiple
			// places where it should be used
			// auto & op = params.get_operator();
			// op.reinitialize_vector(predict);

			auto & sol = getvec<workvec::current_sol>();
			sol.copy(predict);
		}
		else {
			// use previous timestep as initial guess
			auto & sol = getvec<workvec::current_sol>();
			sol.copy(prev[0].solution);
		}

		compute_source_term();
	}

	void compute_source_term() {
		auto & f = getvec<workvec::source>();

		const auto current_integrator = get_current_integrator();
		std::vector<double> h(memory_size(current_integrator));
		h[0] = current_dt;
		for (int i = 1; i < memory_size(current_integrator); ++i) {
			h[i] = h[i - 1] + prev[i - 1].dt;
		}

		double h1, h2, h3, h4, h5, h6;
		double n1, n2, n3, n4, n5, n6;
		double a0, a1, a2, a3, a4, a5, a6;
		switch (current_integrator) {
			case method::be: {
				f.scale(-1., prev[0].solution);
				gamma = current_dt;
				break;
			}
			case method::cn: {
				double eps = 0.0;
				double alpha = 0.5 * current_dt + eps;
				double beta = 0.5 * current_dt - eps;

				const auto & prev_func = getvec<workvec::previous_function>();

				// set source term to -(u_n + (dt / 2 - \eps)f(u_n))
				f.axpy(beta, prev_func, prev[0].solution);
				f.scale(-1.);

				gamma = alpha;
				break;
			}
			case method::bdf2: {
				h1 = h[0];
				h2 = h[1];
				alpha = h1 + h2;
				a0 = (h1 * h2) / alpha;
				a1 = -(h2 * h2) / (alpha * (h2 - h1));
				a2 = -(h1 * h1) / (alpha * (h1 - h2));

				f.scale(a1, prev[0].solution);
				f.axpy(a2, prev[1].solution, f);

				gamma = a0;
				break;
			}
			case method::bdf3: {
				h1 = h[0];
				h2 = h[1];
				h3 = h[2];
				n1 = h2 * h3;
				n2 = h3 * h1;
				n3 = h1 * h2;
				alpha = n1 + n2 + n3;
				a0 = (h1 * h2 * h3) / alpha;
				a1 = -(n1 * n1) / (alpha * (h2 - h1) * (h3 - h1));
				a2 = -(n2 * n2) / (alpha * (h1 - h2) * (h3 - h2));
				a3 = -(n3 * n3) / (alpha * (h1 - h3) * (h2 - h3));

				f.scale(a1, prev[0].solution);
				f.axpy(a2, prev[1].solution, f);
				f.axpy(a3, prev[2].solution, f);

				gamma = a0;
				break;
			}
			case method::bdf4: {
				h1 = h[0];
				h2 = h[1];
				h3 = h[2];
				h4 = h[3];
				n1 = h2 * h3 * h4;
				n2 = h3 * h4 * h1;
				n3 = h4 * h1 * h2;
				n4 = h1 * h2 * h3;
				alpha = n1 + n2 + n3 + n4;

				a0 = (h1 * h2 * h3 * h4) / alpha;
				a1 = -(n1 * n1) / (alpha * (h2 - h1) * (h3 - h1) * (h4 - h1));
				a2 = -(n2 * n2) / (alpha * (h1 - h2) * (h3 - h2) * (h4 - h2));
				a3 = -(n3 * n3) / (alpha * (h1 - h3) * (h2 - h3) * (h4 - h3));
				a4 = -(n4 * n4) / (alpha * (h1 - h4) * (h2 - h4) * (h3 - h4));

				f.scale(a1, prev[0].solution);
				f.axpy(a2, prev[1].solution, f);
				f.axpy(a3, prev[2].solution, f);
				f.axpy(a4, prev[3].solution, f);

				gamma = a0;
				break;
			}
			case method::bdf5: {
				h1 = h[0];
				h2 = h[1];
				h3 = h[2];
				h4 = h[3];
				h5 = h[4];
				n1 = h2 * h3 * h4 * h5;
				n2 = h3 * h4 * h5 * h1;
				n3 = h4 * h5 * h1 * h2;
				n4 = h5 * h1 * h2 * h3;
				n5 = h1 * h2 * h3 * h4;
				alpha = n1 + n2 + n3 + n4 + n5;

				a0 = (h1 * h2 * h3 * h4 * h5) / alpha;
				a1 = -(n1 * n1) /
				     (alpha * (h2 - h1) * (h3 - h1) * (h4 - h1) * (h5 - h1));
				a2 = -(n2 * n2) /
				     (alpha * (h1 - h2) * (h3 - h2) * (h4 - h2) * (h5 - h2));
				a3 = -(n3 * n3) /
				     (alpha * (h1 - h3) * (h2 - h3) * (h4 - h3) * (h5 - h3));
				a4 = -(n4 * n4) /
				     (alpha * (h1 - h4) * (h2 - h4) * (h3 - h4) * (h5 - h4));
				a5 = -(n5 * n5) /
				     (alpha * (h1 - h5) * (h2 - h5) * (h3 - h5) * (h4 - h5));

				f.scale(a1, prev[0].solution);
				f.axpy(a2, prev[1].solution, f);
				f.axpy(a3, prev[2].solution, f);
				f.axpy(a4, prev[3].solution, f);
				f.axpy(a5, prev[4].solution, f);

				gamma = a0;
				break;
			}
			case method::bdf6: {
				h1 = h[0];
				h2 = h[1];
				h3 = h[2];
				h4 = h[3];
				h5 = h[4];
				h6 = h[5];

				n1 = h2 * h3 * h4 * h5 * h6;
				n2 = h3 * h4 * h5 * h6 * h1;
				n3 = h4 * h5 * h6 * h1 * h2;
				n4 = h5 * h6 * h1 * h2 * h3;
				n5 = h6 * h1 * h2 * h3 * h4;
				n6 = h1 * h2 * h3 * h4 * h5;
				alpha = n1 + n2 + n3 + n4 + n5 + n6;

				a0 = (h1 * h2 * h3 * h4 * h5 * h6) / alpha;
				a1 = -(n1 * n1) / (alpha * (h2 - h1) * (h3 - h1) * (h4 - h1) *
				                   (h5 - h1) * (h6 - h1));
				a2 = -(n2 * n2) / (alpha * (h1 - h2) * (h3 - h2) * (h4 - h2) *
				                   (h5 - h2) * (h6 - h2));
				a3 = -(n3 * n3) / (alpha * (h1 - h3) * (h2 - h3) * (h4 - h3) *
				                   (h5 - h3) * (h6 - h3));
				a4 = -(n4 * n4) / (alpha * (h1 - h4) * (h2 - h4) * (h3 - h4) *
				                   (h5 - h4) * (h6 - h4));
				a5 = -(n5 * n5) / (alpha * (h1 - h5) * (h2 - h5) * (h3 - h5) *
				                   (h4 - h5) * (h6 - h5));
				a6 = -(n6 * n6) / (alpha * (h1 - h5) * (h2 - h6) * (h3 - h6) *
				                   (h4 - h6) * (h5 - h6));

				f.scale(a1, prev[0].solution);
				f.axpy(a2, prev[1].solution, f);
				f.axpy(a3, prev[2].solution, f);
				f.axpy(a4, prev[3].solution, f);
				f.axpy(a5, prev[4].solution, f);
				f.axpy(a6, prev[5].solution, f);

				gamma = a0;
			}
		}

		// add in time dependent source, g, for problems of the form u_t =
		// f(u)+g or for MGRIT the FAS correction if (params.has_source_term) {
		// 	auto & tdep_source = params.get_source();
		// 	f.axpy(-gamma, tdep_source, f);
		// }

		auto & time_op = params.get_operator();
		time_op.set_scaling(gamma);
	}

	/**
	   Use the approach suggested in Gresho and Sani, Pg 267 to estimate
	   what the time derivative is for CN
	*/
	void estimate_cn_time_derivative() {
		double alpha = -1.;
		double beta = 2. / current_dt;
		double gamma = -1.;

		auto & time_deriv = getvec<workvec::time_deriv>();
		auto & curr_sol = getvec<workvec::current_sol>();
		auto & prev_func = getvec<workvec::previous_function>();

		time_deriv.axpy(alpha, prev[0].solution, curr_sol);
		time_deriv.linear_sum(beta, time_deriv, gamma, prev_func);
	}

	/**
	   Use the approach suggested in Gresho and Sani, Pg 267 to estimate
	   what the time derivative is for CN
	*/
	void estimate_be_time_derivative() {
		double alpha = -1.;
		double beta = 1. / current_dt;

		auto & time_deriv = getvec<workvec::time_deriv>();
		auto & curr_sol = getvec<workvec::current_sol>();

		time_deriv.axpy(alpha, prev[0].solution, curr_sol);
		time_deriv.scale(beta);
	}

	/**
	 we use the approach suggested in Gresho and Sani, Pg 805 to estimate
	 what the time derivative is
	*/
	void estimate_bdf2_time_derivative() {
		double dtt = current_dt / old_dt;
		double alpha = (2. * dtt + 1.) / ((dtt + 1.) * current_dt);
		double beta = -(dtt + 1.) / current_dt;
		double gamma = (dtt * dtt) / ((dtt + 1.) * current_dt);

		auto & time_deriv = getvec<workvec::time_deriv>();
		auto & curr_sol = getvec<workvec::current_sol>();
		time_deriv.linear_sum(alpha, curr_sol, beta, prev[0].solution);
		time_deriv.axpy(gamma, prev[1].solution, time_deriv);
	}

	void estimate_time_derivative() {
		auto current_integrator = get_current_integrator();
		switch (current_integrator) {
			case method::cn:
				estimate_cn_time_derivative();
				break;
			case method::be:
				estimate_be_time_derivative();
				break;
			default: // bdf2-6
				estimate_bdf2_time_derivative();
		}
	}

	void evaluate_predictor() {
		switch (params.integrator) {
			case method::cn:
				if (params.predictor == predictor::ab2) {
					evaluate_ab2_predictor();
				}
				else {
					flog_error(
						"Crack-Nicolson currently only supports ab2 predictor");
				}
				break;
			case method::be:
				evaluate_forward_euler_predictor();
				break;
			default:
				if (first_step) {
					if (params.starting_integrator == method::cn ||
					    params.starting_integrator == method::be) {
						evaluate_forward_euler_predictor();
					}
					else {
						flog_error(
							"ERROR: starting integrator must be CN or BE");
					}
				}
				else {
					if (params.predictor == predictor::leapfrog) {
						evaluate_leapfrog_predictor();
					}
					else {
						flog_error("BDF2-6 currently only supports the "
						           "leapfrog predcictor");
					}
				}
		}

		auto & predict = getvec<workvec::predict>();
		auto & op = params.get_operator();
		if (!op.is_valid(predict)) {
			// constant extrapolation in time for the predictor
			evaluate_forward_euler_predictor();
		}
	}

	void evaluate_ab2_predictor() {
		double dt_ratio = current_dt / old_dt;
		double alpha = current_dt * (2. + dt_ratio) / 2.;
		double beta = -current_dt * dt_ratio / 2.;

		auto & predict = getvec<workvec::predict>();
		auto & curr_func = getvec<workvec::current_function>();
		auto & old_td = getvec<workvec::old_td>();

		predict.linear_sum(alpha, curr_func, beta, old_td);
		predict.add(predict, prev[0].solution);
	}

	void evaluate_leapfrog_predictor() {
		double dt_ratio = current_dt / old_dt;
		double alpha = dt_ratio * dt_ratio;
		double beta = 1. - alpha;
		double gamma = (1. + dt_ratio) * current_dt;

		auto & predict = getvec<workvec::predict>();
		auto & time_deriv = getvec<workvec::time_deriv>();

		predict.linear_sum(alpha, prev[1].solution, beta, prev[0].solution);
		predict.axpy(gamma, time_deriv, predict);
	}

	void evaluate_forward_euler_predictor() {
		// double alpha = 0.;
		// double beta = 1. - alpha;
		double gamma = current_dt;

		if (first_step) {
			auto time_op = params.get_operator();
			auto & scratch = getvec<workvec::scratch>();
			auto & curr_func = getvec<workvec::current_function>();
			auto & time_deriv = getvec<workvec::time_deriv>();

			scratch.copy(prev[0].solution);
			time_op.apply_rhs(scratch, curr_func);
			time_deriv.copy(curr_func);
		}

		auto & predict = getvec<workvec::predict>();
		auto & time_deriv = getvec<workvec::time_deriv>();

		predict.copy(prev[0].solution);
		predict.axpy(gamma, time_deriv, predict);
	}

	/*
	  Estimate dynamical time scale.  Implements time step control that limits
	  relative change in the solution.
	*/
	double estimate_dynamical_time_scale(double curr_dt) {
		if (params.integrator == method::cn) {
			flog_error("Not implemented");
		}

		auto & scratch = getvec<workvec::scratch>();

		scratch.add_scalar(prev[0].solution,
		                   std::numeric_limits<double>::epsilon());
		scratch.divide(prev[1].solution, scratch);
		scratch.add_scalar(scratch, -1);
		scratch.abs(scratch);

		std::vector<double> relative_change_in_vars;
		auto & scratch_func = getvec<workvec::scratch_function>();
		scratch_func.apply([&](const auto &... v) {
			if (params.time_trunc_err_norm == vec::norm_type::inf)
				(relative_change_in_vars.push_back(v.inf_norm().get()), ...);
			else
				(relative_change_in_vars.push_back(v.l2norm().get()), ...);
		});

		auto actual_relative_change = std::max_element(
			relative_change_in_vars.begin(), relative_change_in_vars.end());

		double factor = (curr_dt > 0.5) ? 1.05 : 1.1;
		factor = (curr_dt > 10.) ? 1.03 : factor;

		// this is the factor used in Dana's paper
		double cfl_new_dt = std::sqrt(params.target_relative_change /
		                              (*actual_relative_change)) *
		                    curr_dt;
		curr_dt = std::min(cfl_new_dt, factor * curr_dt);

		return curr_dt;
	}

	double estimate_dt_with_truncation_error_estimates(double current_dt,
	                                                   bool good_solution) {
		double dt_factor = 0.;

		// exponent for truncation error
		double p = static_cast<double>(order(get_current_integrator()));

		/*
		  When the PI controller is being used the code will force a switch to
		  the deadbeat controller if any of the following happens:
		  - There is a failure in the nonlinear solution process
		  - There were previous successive rejections of the timestep
		*/
		auto eps = std::numeric_limits<double>::epsilon();
		if (params.use_pi_controller && good_solution &&
		    (!prev_successive_rejects)) {
			// the safety factor limits tries to pull the computed/predicted
			// timestep a bit back
			double safety_factor = 0.8;
			switch (params.pi_controller_type) {
				case controller::H211b: {
					double b = 4.;
					dt_factor = std::pow(safety_factor /
					                         std::max(time_trunc_err_est, eps),
					                     1. / (b * (p + 1.)));
					dt_factor *= std::pow(
						safety_factor / std::max(prev_time_trunc_err_est, eps),
						1. / (b * (p + 1.)));
					dt_factor *= std::pow(alpha, -0.25);
					break;
				}
				case controller::pc4_7: {
					dt_factor = std::pow(safety_factor /
					                         std::max(time_trunc_err_est, eps),
					                     0.4 / (p + 1.));
					dt_factor *=
						std::pow(time_err_est_ratio, 0.7 / (p + 1.)) * alpha;
					break;
				}
				case controller::pc11: {
					dt_factor = std::pow(safety_factor /
					                         std::max(time_trunc_err_est, eps),
					                     1.0 / (p + 1.));
					dt_factor *=
						std::pow(time_err_est_ratio, 1. / (p + 1.)) * alpha;
					break;
				}
				case controller::deadbeat: {
					dt_factor = std::pow(safety_factor /
					                         std::max(time_trunc_err_est, eps),
					                     1. / (p + 1.));
					break;
				}
			}
		}
		else {
			/* The safety factor for the deadbet when recovering from time step
			   rejects should be very conservative */
			double safety_factor = 0.8;
			dt_factor =
				std::pow(safety_factor / std::max(time_trunc_err_est, eps),
			             1. / (p + 1.));
		}

		/* compute how far away from 1 we are, if it's a small fraction away
		   from 1 (in this case 0.1) don't change the timestep. */
		if (params.control_timestep_variation && !params.use_pi_controller) {
			double dt_factor_var = std::fabs(1. - dt_factor);
			if (dt_factor_var < 0.1)
				dt_factor = 1.;
		}

		current_dt = current_dt * dt_factor;

		return current_dt;
	}

	double calculate_LTE_scaling_factor() {
		double error_factor = 0.;

		if (first_step) {
			if (params.starting_integrator == method::be)
				error_factor = 0.5;
			else if (params.starting_integrator == method::cn) {
				// Trompert and Verwer approach
				error_factor =
					1.5; // fairly arbitrary factor, used because the error
				         // estimator is coarse and typically under-estimates
				error_factor *= current_dt;
			}
		}
		else {
			// check!! the expressions given in Gresho and Sani appear to be
			// wrong!
			switch (params.predictor) {
				case predictor::leapfrog:
					// there might be an error here!!
					//      errorFactor = 1.0/((1+alpha)*(1.0+ pow
					//      (alpha/(1.0+alpha), 2.0 ) ) );
					error_factor = (1. + alpha) / (2. + 3. * alpha);
					break;
				case predictor::ab2:
					// compute error factor from M. Pernice communication for
					// AB2 and BDF2
					error_factor = 2. / (6. - 1. / ((alpha + 1) * (alpha + 1)));
					break;
			}
		}

		return error_factor;
	}

	template<class X, class Y, std::size_t N = X::num_components>
	void
	calculate_scaled_LTE_norm(X & x, Y & y, std::array<double, N> & norms) {
		auto & scratch = getvec<workvec::scratch>();
		switch (params.time_error_scaling) {
			case error_scaling::fixed_resolution:
				scratch.scale(params.time_rtol, x);
				scratch.add_scalar(scratch, params.time_atol);
				break;
			case error_scaling::fixed_scaling:
				scratch.zero();

				std::size_t i{0};
				vec::apply(
					[&](const auto & sol_vec, auto & scratch_vec) {
						scratch_vec.add_scalar(sol_vec,
					                           params.problem_scales[i++]);
					},
					x,
					scratch);

				scratch.scale(params.time_rtol);
				break;
		}

		auto & scratch_func = getvec<workvec::scratch_function>();
		scratch_func.subtract(x, y);
		scratch_func.divide(scratch_func, scratch);

		std::size_t i{0};
		vec::apply(
			[this, &i, &norms](const auto & comp) {
				if (params.time_trunc_err_norm == vec::norm_type::inf)
					norms[i++] = comp.inf_norm().get();
				else
					norms[i++] = comp.l2norm().get();
			},
			scratch_func);
	}

	void calculate_temporal_truncation_error() {
		auto & sol_vec = getvec<workvec::current_sol>();
		constexpr auto num_components =
			std::remove_reference_t<decltype(sol_vec)>::num_components;
		std::array<double, num_components> trunc_err_est;
		trunc_err_est.fill(1.);

		if ((integrator_step > 0) ||
		    (first_step && params.use_initial_predictor)) {
			/*
			  Compute a new time step based on truncation error estimates
			  One of the truncation error estimate comes from a private
			  communication with M. Pernice and is based on an AB2 predictor and
			  BDF2 corrector
			*/
			alpha = current_dt / old_dt;
			double error_factor = 0.;

			error_factor = calculate_LTE_scaling_factor();

			std::array<double, num_components> err_norm;
			err_norm.fill(0.);

			auto & y =
				(first_step && (params.starting_integrator == method::cn))
					? prev[0].solution
					: getvec<workvec::predict>();
			calculate_scaled_LTE_norm(sol_vec, y, err_norm);
			for (std::size_t i = 0; i < num_components; ++i) {
				trunc_err_est[i] = err_norm[i] * error_factor;
			}

			// the time error ratio is only used for a PI based controller
			prev_time_trunc_err_est = time_trunc_err_est;
			// store the truncation error estimate in time over E and T
			// it will be used in the get_next_dt method
			time_trunc_err_est =
				*(std::max_element(trunc_err_est.begin(), trunc_err_est.end()));
			time_trunc_err_est = std::max(
				std::numeric_limits<double>::epsilon(), time_trunc_err_est);

			// the time error ratio is only used for a PI based controller
			time_err_est_ratio = prev_time_trunc_err_est / time_trunc_err_est;
		}
		else {
			// for the fist time step pretend we met the truncation error
			// estimate
			prev_time_trunc_err_est = 1.;
			time_trunc_err_est = 1.;
			time_err_est_ratio = 1.;
		}
	}

	template<workvec wvec>
	auto & getvec() {
		return std::get<wvec>(params.work);
	}

	struct history;

	bool prev_successive_rejects;
	int current_steprejects;
	double new_time;

	double time_trunc_err_est;
	double prev_time_trunc_err_est;
	double alpha;
	double time_err_est_ratio;

	bool first_step;
	double gamma;
	history prev;

	bool solver_success;

	int total_steprejects;
};
template<class O, class W, class S>
integrator(parameters<O, W, S>) -> integrator<O, W, S>;

template<class O, class W, class S>
struct integrator<O, W, S>::history {
	struct vec_arr {
		using size_type = std::size_t;
		using vec_t = typename std::remove_reference_t<W>::value_type;
		vec_arr(size_type o, size_type l) : offset(o), len(l) {}
		flecsi::util::span<vec_t> span(W & work) const {
			return {work.data() + offset, len};
		}
		size_type size() const { return len; }

	protected:
		size_type offset;
		size_type len;
	};
	using size_type = typename vec_arr::size_type;

	struct value_type {
		typename vec_arr::vec_t & solution;
		double & dt;
	};

	history(size_type mlen, std::reference_wrapper<W> work)
		: vecs(workvec::nwork, work_size - workvec::nwork), pos(0), len(mlen),
		  work(work) {
		timesteps.resize(mlen);
	}

	value_type operator[](size_type index) {
		size_t beg = len - 1 - pos;
		size_t ind = (beg + index + 1) % len;
		ind = len - 1 - ind;

		flog_assert(ind < len, "Index out of bounds");

		auto span = vecs.span(work);
		return {span[ind], timesteps[ind]};
	}

	template<class V>
	void push(const V & vec, double dt) {
		auto new_el = (*this)(pos++);

		new_el.solution.copy(vec);
		new_el.dt = dt;

		pos %= len;
	}

protected:
	value_type operator()(size_type ind) {
		flog_assert(ind < len, "Index out of bounds");

		auto span = vecs.span(work);
		return {span[ind], timesteps[ind]};
	}

	vec_arr vecs;
	std::vector<double> timesteps;
	size_type pos, len;
	std::reference_wrapper<W> work;
};
}
#endif
