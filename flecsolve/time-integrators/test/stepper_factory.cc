#include <cmath>
#include <utility>

#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/time-integrators/factory.hh"
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

int stepper_test(flecsi::scheduler & scheduler) {
	using namespace time_integrator;
	UNIT () {
		testmesh::ptr mesh;
		auto & topology = init_mesh(scheduler, 1, mesh);
		auto x = vec::make(topology)(xd);
		auto next = vec::make(topology)(xnewd);
		op::core<rate> rhs(rate{});
		auto [settings23, settings45] =
			read_config("stepper-factory.cfg",
		                stepper_factory::options("rk23"),
		                stepper_factory::options("rk45"));

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
	};
}

flecsi::util::unit::driver<stepper_test> driver;
}
}
