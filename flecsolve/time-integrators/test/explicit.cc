#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/time-integrators/rk23.hh"
#include "flecsolve/time-integrators/rk45.hh"
#include "flecsolve/time-integrators/factory.hh"
#include "flecsolve/util/config.hh"

#include <sstream>

#include "flecsolve/util/test/mesh.hh"

namespace flecsolve {

const flecsi::field<double>::definition<testmesh, testmesh::cells> xd, xnewd;

struct parameters {
	double lambda;
};
struct rate : op::base<parameters> {
	using base = op::base<parameters>;
	explicit rate(double l) : base{l} {}
	template<class D, class R>
	void apply(const D & x, R & y) const {
		y.scale(params.lambda, x);
	}
};

int extest(flecsi::scheduler & s) {
	using namespace flecsolve::time_integrator;

	UNIT () {
		testmesh::ptr mptr;

		double ic = 3.;

		auto & msh = init_mesh(s, 1, mptr);

		op::core<rate> F(-1.);

		auto x = vec::make(msh)(xd);
		auto xnew = vec::make(msh)(xnewd);

		auto [ti23_var, ti23_fixed] = std::apply(
			[&](auto &&... s) {
				return std::make_tuple(rk23::integrator(
					rk23::parameters(s, op::ref(F), rk23::make_work(x)))...);
			},
			read_config("explicit.cfg",
		                rk23::options("variable"),
		                rk23::options("fixed")));

		auto [ti45_var, ti45_fixed] = std::apply(
			[&](auto &&... s) {
				return std::make_tuple(rk45::integrator(
					rk45::parameters(s, op::ref(F), rk45::make_work(x)))...);
			},
			read_config("explicit.cfg",
		                rk45::options("variable"),
		                rk45::options("fixed")));

		auto run = [&](auto & ti) {
			x.set_scalar(ic);
			auto dt = ti.get_current_dt();
			int rejects = 0;
			while (ti.steps_remaining() && ti.get_current_time() < ti.get_final_time()) {
				ti.advance(dt, x, xnew);
				auto good_solution = ti.check_solution();
				if (good_solution or ti.fixed_dt()) {
					ti.update();
					std::swap(x, xnew);
				}
				else
					++rejects;
				dt = ti.get_next_dt(good_solution);
			}

			auto sol =
				ic * std::exp(F.get_params().lambda * ti.get_final_time());
			auto approx = x.max().get();
			return std::tuple(ti.get_current_time(),
			                  std::abs(sol - approx),
			                  ti.get_current_step(), rejects);
		};
		{
			auto ans = run(ti23_fixed);
			EXPECT_EQ(std::get<0>(ans), 1.0);
			EXPECT_LT(std::get<1>(ans), 1e-5);
		}
		{
			auto ans = run(ti45_fixed);
			EXPECT_EQ(std::get<0>(ans), 1.0);
			EXPECT_LT(std::get<1>(ans), 1e-9);
		}
		{
			auto [end, err, step, rejects] = run(ti23_var);
			EXPECT_EQ(end, 1.0);
			EXPECT_LT(err, 1e-5);
			EXPECT_EQ(step, 32);
			EXPECT_GT(rejects, 0);
		}
		{
			auto [end, err, step, rejects] = run(ti45_var);
			EXPECT_EQ(end, 1.0);
			EXPECT_LT(err, 1e-6);
			EXPECT_EQ(step, 6);
			EXPECT_GT(rejects, 0);
		}
		{
			auto settings = read_config("integrator-factory.cfg",
				integrator_factory::options("rk23-variable"),
				integrator_factory::options("rk23-fixed"),
				integrator_factory::options("rk45-variable"),
				integrator_factory::options("rk45-fixed"));

			// Use the same fixed/adaptive decay cases as direct construction.
			// Exact agreement checks both selection and rejected-step retries.
			auto compare = [&](const auto & factory_settings, auto direct) {
				auto ti = integrator_factory::make(factory_settings, x, op::ref(F));
				const auto & view = ti;
				EXPECT_EQ(view.get_current_time(), 0.);
				EXPECT_EQ(view.get_final_time(), 1.);
				EXPECT_EQ(view.get_current_step(), 0);
				EXPECT_EQ(view.get_current_dt(), direct.get_current_dt());
				EXPECT_EQ(view.fixed_dt(), direct.fixed_dt());
				EXPECT_TRUE(view.steps_remaining());
				auto expected = run(direct);
				auto actual = run(ti);
				EXPECT_EQ(std::get<0>(actual), std::get<0>(expected));
				EXPECT_EQ(std::get<1>(actual), std::get<1>(expected));
				EXPECT_EQ(std::get<2>(actual), std::get<2>(expected));
				EXPECT_EQ(std::get<3>(actual), std::get<3>(expected));
				EXPECT_EQ(ti.num_step_rejects(), direct.num_step_rejects());
			};
			auto variable23 = read_config("explicit.cfg",
				rk23::options("variable"), rk23::options("fixed"));
			auto variable45 = read_config("explicit.cfg",
				rk45::options("variable"), rk45::options("fixed"));
			compare(std::get<0>(settings), rk23::integrator(rk23::parameters(
				std::get<0>(variable23), op::ref(F), rk23::make_work(x))));
			compare(std::get<1>(settings), rk23::integrator(rk23::parameters(
				std::get<1>(variable23), op::ref(F), rk23::make_work(x))));
			compare(std::get<2>(settings), rk45::integrator(rk45::parameters(
				std::get<0>(variable45), op::ref(F), rk45::make_work(x))));
			compare(std::get<3>(settings), rk45::integrator(rk45::parameters(
				std::get<1>(variable45), op::ref(F), rk45::make_work(x))));

			// Shared operator handles follow the same construction path.
			auto shared = op::make_shared<rate>(-1.);
			auto limited = std::get<1>(settings);
			std::get<rk23::settings>(limited.target_settings).max_steps = 1;
			auto ti = integrator_factory::make(limited, x, shared);
			x.set_scalar(ic);
			ti.advance(ti.get_current_dt(), x, xnew);
			ti.update();
			EXPECT_EQ(ti.get_current_step(), 1);
			EXPECT_FALSE(ti.steps_remaining());
		}
		{
			method_target target;
			std::istringstream valid23("rk23"), valid45("rk45"), validbdf("bdf"),
				invalid("unknown");
			valid23 >> target;
			EXPECT_FALSE(valid23.fail());
			EXPECT_TRUE(target == method_target::rk23);
			valid45 >> target;
			EXPECT_FALSE(valid45.fail());
			EXPECT_TRUE(target == method_target::rk45);
			validbdf >> target;
			EXPECT_FALSE(validbdf.fail());
			EXPECT_TRUE(target == method_target::bdf);
			invalid >> target;
			EXPECT_TRUE(invalid.fail());
		}

		{
			auto dt = 0.05;

			rk23::stepper rk23_double(
				rk23::parameters(op::ref(F), rk23::make_work(x)));
			x.set_scalar(ic);
			rk23_double.advance(dt, x, xnew);
			auto rk23_double_ans = xnew.max().get();

			rk23::stepper rk23_future(
				rk23::parameters(op::ref(F), rk23::make_work(x)));
			x.set_scalar(ic);
			rk23_future.advance(flecsi::make_future(dt), x, xnew);
			EXPECT_EQ(xnew.max().get(), rk23_double_ans);

			rk45::stepper rk45_double(
				rk45::parameters(op::ref(F), rk45::make_work(x)));
			x.set_scalar(ic);
			rk45_double.advance(dt, x, xnew);
			auto rk45_double_ans = xnew.max().get();

			rk45::stepper rk45_future(
				rk45::parameters(op::ref(F), rk45::make_work(x)));
			x.set_scalar(ic);
			rk45_future.advance(flecsi::make_future(dt), x, xnew);
			EXPECT_EQ(xnew.max().get(), rk45_double_ans);
		}
	};
}

flecsi::util::unit::driver<extest> driver;
}
