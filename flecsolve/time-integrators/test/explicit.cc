#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/time-integrators/rk23.hh"
#include "flecsolve/time-integrators/rk45.hh"
#include "flecsolve/util/config.hh"

#include "flecsolve/util/test/mesh.hh"

namespace flecsolve {

const flecsi::field<double>::definition<testmesh, testmesh::cells> xd, xnewd;

struct parameters { double lambda; };
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
				return std::make_tuple(rk23::integrator(rk23::parameters(
					                                        s, op::ref(F), rk23::make_work(x)))...);
			},
			read_config("explicit.cfg",
		                rk23::options("variable"),
		                rk23::options("fixed")));

		auto [ti45_var, ti45_fixed] = std::apply(
			[&](auto &&... s) {
				return std::make_tuple(rk45::integrator(rk45::parameters(
					                                        s, op::ref(F), rk45::make_work(x)))...);
			},
			read_config("explicit.cfg",
		                rk45::options("variable"),
		                rk45::options("fixed")));

		auto run = [&](auto & ti) {
			x.set_scalar(ic);
			auto dt = ti.get_current_dt();
			while (ti.get_current_time() < ti.get_final_time()) {
				ti.advance(dt, x, xnew);
				auto good_solution = ti.check_solution();
				if (good_solution or ti.fixed_dt()) {
					ti.update();
					std::swap(x, xnew);
				}
				dt = ti.get_next_dt(good_solution);
			}

			auto sol = ic * std::exp(F.get_params().lambda * ti.get_final_time());
			auto approx = x.max().get();
			return std::tuple(ti.get_final_time(),
			                  std::abs(sol - approx),
			                  ti.get_current_step());
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
			auto [end, err, step] = run(ti23_var);
			EXPECT_EQ(end, 1.0);
			EXPECT_LT(err, 1e-5);
			EXPECT_EQ(step, 32);
		}
		{
			auto [end, err, step] = run(ti45_var);
			EXPECT_EQ(end, 1.0);
			EXPECT_LT(err, 1e-6);
			EXPECT_EQ(step, 6);
		}
	};
}

flecsi::util::unit::driver<extest> driver;
}
