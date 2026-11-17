#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/util/config.hh"
#include "flecsolve/time-integrators/factory.hh"

#include <stdexcept>

#include "flecsolve/util/test/mesh.hh"
#include "flecsolve/operators/core.hh"

namespace flecsolve {

const flecsi::field<double>::definition<testmesh, testmesh::cells> xd, xnewd;

struct parameters {
	double lambda;
	double gamma;
};

struct rate : op::base<parameters> {
	rate(double lambda) :
		op::base<parameters>(parameters{lambda, 1.}) {}

	template<class D, class R>
	void residual(const D & b, const R & x, R & r) const {
		apply(x, r);
		r.subtract(b, r);
	}

	template<class D, class R>
	void apply(const D & x, R & y) const {
		// f(x^{n+1})
		apply_rhs(x, y);
		// y = x^{n+1} - scaling * f(x^{n+1})
		y.axpy(-params.gamma, y, x);
	}

	template<class D, class R>
	void apply_rhs(const D & x, R & y) const {
		y.scale(params.lambda, x);
	}

	template<class V>
	bool is_valid(const V &) {
		return true;
	}

	double get_scaling() const { return params.gamma; }
	void set_scaling(double scaling) { params.gamma = scaling; }

	double get_rate() const { return params.lambda; }
};

double get_rate(const rate & r) { return r.get_rate(); }

struct rate_solver : op::base<> {
	rate_solver(op::handle<op::core<rate>> oph, bool fail_once = false)
		: F(oph), fail_once(fail_once) {}

	template<class D, class R>
	solve_info apply(const D & b, R & x) const {
		if (fail_once) {
			fail_once = false;
			solve_info info;
			info.status = solve_info::stop_reason::diverged_iters;
			return info;
		}
		auto rhs = b.min().get();
		const auto & op = F.get();
		auto sol = rhs / (1. - op.get_rate() * op.get_scaling());
		x.set_scalar(sol);

		solve_info info;
		info.status = solve_info::stop_reason::converged_atol;
		return info;
	}

	op::handle<op::core<rate>> F;
	mutable bool fail_once;
};

int bdftest(flecsi::scheduler & s) {
	using namespace flecsolve::time_integrator;

	UNIT () {
		testmesh::ptr mptr;

		double ic = 3.;

		auto & msh = init_mesh(s, 1, mptr);

		auto F = op::make_shared<rate>(-1.);
		auto x = vec::make(msh)(xd);
		auto xnew = vec::make(msh)(xnewd);

		auto solver = op::make_shared<rate_solver>(F);
		auto [bdf2_settings, bdf5_settings, stepper_settings] = read_config(
			"implicit.cfg",
			bdf::options("bdf-2"),
			bdf::options("bdf-5"),
			bdf::stepper_options("bdf-stepper"));
		auto [ti2, ti5] = std::apply(
			[&](auto &&... s) {
				return std::make_tuple(bdf::integrator(bdf::parameters(
					                                       s, F, bdf::make_work(x), solver))...);
			},
			std::tie(bdf2_settings, bdf5_settings));

		auto run = [&](auto & ti) {
			x.set_scalar(ic);
			auto dt = ti.get_current_dt();
			std::cout.precision(10);
			while (ti.steps_remaining() &&
			       ti.get_current_time() < ti.get_final_time()) {
				ti.advance(dt, x, xnew);
				auto good_solution = ti.check_solution();
				if (good_solution) {
					ti.update();
					std::swap(x, xnew);
				}
				dt = ti.get_next_dt(good_solution);
			}

			auto sol =
				ic * std::exp(get_rate(F) * ti.get_final_time());
			auto approx = x.max().get();
			return std::tuple(ti.get_current_time(),
			                  std::abs(sol - approx),
			                  ti.get_current_step(),
			                  ti.num_step_rejects());
		};
		{
			auto info = run(ti2);
			EXPECT_EQ(std::get<0>(info), 1.);
			EXPECT_LT(std::get<1>(info), 1e-3);
			EXPECT_LE(std::get<2>(info), 49);
			EXPECT_LE(std::get<3>(info), 7);
		}
		{
			auto info = run(ti5);
			EXPECT_EQ(std::get<0>(info), 1.);
			EXPECT_LT(std::get<1>(info), 1e-7);
			EXPECT_EQ(std::get<2>(info), 48);
			EXPECT_EQ(std::get<3>(info), 14);
		}
		{
			auto [settings2, settings5] =
				read_config("integrator-factory.cfg",
			                integrator_factory::options("bdf-2"),
			                integrator_factory::options("bdf-5"));
			auto compare = [&](const auto & settings,
			                   const bdf::settings & direct_settings) {
				bdf::integrator direct(bdf::parameters(
					direct_settings, F, bdf::make_work(x), solver));
				auto expected = run(direct);
				auto ti = integrator_factory::make(settings, x, F, solver);
				const auto & view = ti;
				EXPECT_EQ(view.get_current_time(),
				          direct_settings.initial_time);
				EXPECT_EQ(view.get_final_time(), direct_settings.final_time);
				EXPECT_EQ(view.get_current_dt(), direct_settings.initial_dt);
				EXPECT_EQ(view.get_current_step(), 0);
				EXPECT_TRUE(view.steps_remaining());
				EXPECT_FALSE(view.fixed_dt());
				auto actual = run(ti);
				EXPECT_EQ(std::get<0>(actual), std::get<0>(expected));
				EXPECT_EQ(std::get<1>(actual), std::get<1>(expected));
				EXPECT_EQ(std::get<2>(actual), std::get<2>(expected));
				EXPECT_EQ(std::get<3>(actual), std::get<3>(expected));
			};
			compare(settings2, bdf2_settings);
			compare(settings5, bdf5_settings);

			// BE and CN are selected through the existing BDF method settings.
			for (auto method : {bdf::method::be, bdf::method::cn}) {
				auto settings = settings2;
				auto & options =
					std::get<bdf::settings>(settings.target_settings);
				options.integrator = options.starting_integrator = method;
				options.timestep_strategy = bdf::strategy::constant;
				options.use_predictor = options.calculate_time_trunc_error =
					false;
				options.initial_dt = 0.05;
				compare(settings, options);
			}

			// Borrowed handles and a solver failure on the first attempt:
			// startup and the solution must survive rejection and a move.
			op::core<rate_solver> failing_solver(F, true);
			auto ti = integrator_factory::make(
				settings2, x, op::ref(F.get()), op::ref(failing_solver));
			x.set_scalar(ic);
			ti.advance(ti.get_current_dt(), x, xnew);
			EXPECT_FALSE(ti.check_solution());
			EXPECT_EQ(ti.get_current_time(), 0.);
			EXPECT_EQ(ti.get_current_step(), 0);
			EXPECT_EQ(x.max().get(), ic);
			EXPECT_LT(ti.get_next_dt(false), 0.5);
			auto moved = std::move(ti);
			auto retried = run(moved);
			EXPECT_EQ(std::get<0>(retried), 1.);
			EXPECT_LT(std::get<1>(retried), 1e-3);

			auto limited = settings2;
			std::get<bdf::settings>(limited.target_settings).max_steps = 1;
			auto one_step = integrator_factory::make(limited, x, F, solver);
			auto stopped = run(one_step);
			EXPECT_EQ(std::get<2>(stopped), 1);
			EXPECT_FALSE(one_step.steps_remaining());
		}
		{
			// RK selected through the time operator and solver: the RHS comes
			// from apply_rhs(), so this matches the explicit test's decay
			// problem, and the solver is never called.
			auto [rk23_settings, rk45_settings, bdf_settings] =
				read_config("integrator-factory.cfg",
			                integrator_factory::options("rk23-variable"),
			                integrator_factory::options("rk45-variable"),
			                integrator_factory::options("bdf-2"));
			F.get().set_scaling(7.);
			auto unused_solver = op::make_shared<rate_solver>(F, true);
			auto ti23 =
				integrator_factory::make(rk23_settings, x, F, unused_solver);
			auto info23 = run(ti23);
			EXPECT_EQ(std::get<0>(info23), 1.);
			EXPECT_LT(std::get<1>(info23), 1e-5);
			EXPECT_EQ(std::get<2>(info23), 32);
			auto ti45 =
				integrator_factory::make(rk45_settings, x, F, unused_solver);
			auto info45 = run(ti45);
			EXPECT_EQ(std::get<0>(info45), 1.);
			EXPECT_LT(std::get<1>(info45), 1e-6);
			EXPECT_EQ(std::get<2>(info45), 6);
			EXPECT_TRUE(unused_solver.get().fail_once);

			// BDF cannot be built from an RHS handle alone.
			bool missing_solver = false;
			try {
				integrator_factory::make(bdf_settings, x, F);
			}
			catch (const std::invalid_argument &) {
				missing_solver = true;
			}
			EXPECT_TRUE(missing_solver);
		}

		{
			bdf::stepper ti(
				bdf::parameters(stepper_settings, F, bdf::make_work(x), solver));
			x.set_scalar(ic);
			ti.advance(0.5, x, xnew);
			ti.update();

			EXPECT_EQ(ti.get_current_step(), 1);
			EXPECT_EQ(ti.num_step_rejects(), 0);
			EXPECT_LT(std::abs(xnew.max().get() - 2.), 1e-12);
		}
	};
}

flecsi::util::unit::driver<bdftest> driver;
}
