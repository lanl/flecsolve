#include <cmath>
#include <utility>
#include <sstream>

#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/time-integrators/factory.hh"
#include "flecsolve/time-integrators/operator_adapter.hh"
#include "flecsolve/util/test/mesh.hh"
#include "flecsolve/vectors/topo_view.hh"

namespace flecsolve {
namespace {
const flecsi::field<double>::definition<testmesh, testmesh::cells> xd, xnewd;

struct rate : op::base<> {
	template<class D, class R>
	void apply(const D & x, R & y) const {
		y.scale(-1., x);
	}
};

using time_rate = time_integrator::operator_adapter<rate>;

struct rate_solver : op::base<> {
	explicit rate_solver(op::handle<op::core<time_rate>> time_op)
		: time_op(std::move(time_op)) {}

	template<class D, class R>
	solve_info apply(const D & rhs, R & x) const {
		++calls;
		solve_info info;
		if (fail_once) {
			fail_once = false;
			info.status = solve_info::stop_reason::diverged_iters;
		}
		else {
			x.scale(1. / (1. + time_op.get().get_scaling()), rhs);
			info.status = solve_info::stop_reason::converged_atol;
		}
		return info;
	}

	op::handle<op::core<time_rate>> time_op;
	mutable bool fail_once{false};
	mutable int calls{0};
};

int stepper_test(flecsi::scheduler & scheduler) {
	using namespace time_integrator;
	UNIT () {
		testmesh::ptr mesh;
		auto & topology = init_mesh(scheduler, 1, mesh);
		auto x = vec::make(topology)(xd);
		auto next = vec::make(topology)(xnewd);
		op::core<rate> rhs(rate{});
		auto [settings23, settings45, settings_bdf] =
			read_config("stepper-factory.cfg",
		                stepper_factory::options("rk23"),
		                stepper_factory::options("rk45"),
		                stepper_factory::options("bdf"));

		// Repeat steps at varying dt to check the application-controlled path.
		auto run = [&](auto & stepper, bool future_dt) {
			x.set_scalar(3.);
			for (double dt : {0.05, 0.1, 0.025}) {
				if (future_dt)
					stepper.advance(flecsi::make_future(dt), x, next);
				else
					stepper.advance(dt, x, next);
				std::swap(x, next);
			}
			return x.max().get();
		};
		auto check = [&](const auto & settings, auto method, auto direct) {
			auto expected = run(direct, false);
			EXPECT_LT(std::abs(expected - 3. * std::exp(-0.175)), 2e-5);
			auto scalar = stepper_factory::make(settings, x, op::ref(rhs));
			EXPECT_EQ(run(scalar, false), expected);
			auto future = stepper_factory::make(settings, x, op::ref(rhs));
			EXPECT_EQ(run(future, true), expected);

			auto shared =
				stepper_factory::make(method, x, op::make_shared<rate>(rate{}));
			EXPECT_EQ(run(shared, true), expected);
		};
		check(
			settings23,
			stepper_factory::target::rk23,
			rk23::stepper(rk23::parameters(op::ref(rhs), rk23::make_work(x))));
		check(
			settings45,
			stepper_factory::target::rk45,
			rk45::stepper(rk45::parameters(op::ref(rhs), rk45::make_work(x))));
		{
			// A solver-taking call can also select RK. Its RHS view must not
			// apply the implicit system I - gamma F or invoke the solver.
			auto time_op = op::make_shared<time_rate>(time_rate{});
			time_op.get().set_scaling(7.);
			auto solver = op::make_shared<rate_solver>(time_op);
			for (const auto & settings : {settings23, settings45}) {
				auto direct = stepper_factory::make(settings, x, op::ref(rhs));
				auto expected = run(direct, false);
				auto step = stepper_factory::make(settings, x, time_op, solver);
				x.set_scalar(3.);
				for (double dt : {0.05, 0.1, 0.025}) {
					step.advance(flecsi::make_future(dt), x, next);
					EXPECT_TRUE(step.check_solution());
					step.update();
					std::swap(x, next);
				}
				EXPECT_EQ(x.max().get(), expected);
				EXPECT_EQ(step.get_current_step(), 3);
				EXPECT_EQ(step.num_step_rejects(), 0);
			}
			EXPECT_EQ(solver.get().calls, 0);
		}
		{
			auto time_op = op::make_shared<time_rate>(time_rate{});
			auto solver = op::make_shared<rate_solver>(time_op);
			auto run_bdf = [&](auto & step) {
				x.set_scalar(3.);
				solver.get().fail_once = true;
				double time = 0.;
				int rejects = 0;
				bool first_attempt = true;
				for (double dt :
				     {0.01, 0.02, 0.005, 0.015, 0.01, 0.0075, 0.0125}) {
					bool accepted = false;
					for (int attempt = 0; attempt < 64 && !accepted;
					     ++attempt) {
						step.advance(dt, x, next);
						accepted = step.check_solution();
						if (first_attempt) {
							EXPECT_FALSE(accepted);
							EXPECT_EQ(step.get_current_step(), 0);
							EXPECT_EQ(x.max().get(), 3.);
							first_attempt = false;
						}
						if (accepted) {
							step.update();
							std::swap(x, next);
							time += dt;
						}
						else {
							++rejects;
							dt *= 0.5;
						}
					}
					EXPECT_TRUE(accepted);
				}
				EXPECT_EQ(step.get_current_step(), 7);
				EXPECT_EQ(step.num_step_rejects(), rejects);
				EXPECT_LT(std::abs(x.max().get() - 3. * std::exp(-time)), 1e-4);
				return std::tuple(x.max().get(), time, rejects);
			};
			for (auto method : {bdf::method::bdf2, bdf::method::bdf5}) {
				auto settings = settings_bdf;
				auto & options =
					std::get<bdf::stepper_settings>(settings.target_settings);
				options.integrator = method;
				bdf::stepper direct(bdf::parameters(
					options, time_op, bdf::make_work(x), solver));
				auto expected = run_bdf(direct);
				auto original = stepper_factory::make(
					settings, x, op::ref(time_op.get()), op::ref(solver.get()));
				auto step = std::move(original);
				bool future_rejected = false;
				try {
					step.advance(flecsi::make_future(0.01), x, next);
				}
				catch (const std::invalid_argument &) {
					future_rejected = true;
				}
				EXPECT_TRUE(future_rejected);
				auto actual = run_bdf(step);
				EXPECT_EQ(std::get<0>(actual), std::get<0>(expected));
				EXPECT_EQ(std::get<1>(actual), std::get<1>(expected));
				EXPECT_EQ(std::get<2>(actual), std::get<2>(expected));
			}
			bool missing_solver = false;
			try {
				stepper_factory::make(settings_bdf, x, time_op);
			}
			catch (const std::invalid_argument &) {
				missing_solver = true;
			}
			EXPECT_TRUE(missing_solver);

			// A rejected first attempt must not leave its timestep in the
			// startup error estimate: the retry matches a fresh stepper.
			auto first_step = [&](bool reject_first) {
				auto step =
					stepper_factory::make(settings_bdf, x, time_op, solver);
				x.set_scalar(3.);
				if (reject_first) {
					solver.get().fail_once = true;
					step.advance(0.02, x, next);
					EXPECT_FALSE(step.check_solution());
				}
				step.advance(0.01, x, next);
				return std::pair(step.check_solution(), next.max().get());
			};
			auto retried = first_step(true);
			auto fresh = first_step(false);
			EXPECT_EQ(retried.first, fresh.first);
			EXPECT_EQ(retried.second, fresh.second);
		}
		{
			stepper_factory::target target;
			std::istringstream valid("bdf"), invalid("unknown");
			valid >> target;
			EXPECT_FALSE(valid.fail());
			EXPECT_TRUE(target == stepper_factory::target::bdf);
			invalid >> target;
			EXPECT_TRUE(invalid.fail());
		}
	};
}

flecsi::util::unit::driver<stepper_test> driver;
}
}
