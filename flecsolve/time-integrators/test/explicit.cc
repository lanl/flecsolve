#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/time-integrators/rk23.hh"
#include "flecsolve/time-integrators/rk45.hh"
#include "flecsolve/util/config.hh"

#include "flecsolve/solvers/test/csr_utils.hh"

namespace flecsolve {

testmesh::slot msh;
testmesh::cslot coloring;

const flecsi::field<double>::definition<testmesh, testmesh::cells> xd;

struct rate {
	template<class D, class R>
	void apply(const D & x, R & y) {
		y.scale(lambda, x);
	}

	double lambda;
};

int extest() {
	using namespace flecsolve::time_integrator;

	UNIT () {
		double ic = 3.;

		init_mesh(1, msh, coloring);

		rate F{-1};

		vec::mesh x(msh, xd(msh));

		rk23::parameters params23(
			"time-int", std::ref(F), rk23::topo_work<>::get(x));
		rk45::parameters params45(
			"time-int", std::ref(F), rk45::topo_work<>::get(x));
		read_config("explicit.cfg", params23);
		read_config("explicit.cfg", params45);
		rk23::integrator ti23(std::move(params23));
		rk45::integrator ti45(std::move(params45));

		auto run = [&](auto & ti) {
			x.set_scalar(ic);
			while (ti.get_current_time() < ti.get_final_time()) {
				ti.advance(ti.get_current_dt(), x, x);
				ti.update();
			}
			auto sol = ic * std::exp(F.lambda * ti.get_final_time());
			auto approx = x.max().get();
			return std::pair(ti.get_final_time(), std::abs(sol - approx));
		};
		{
			auto ans = run(ti23);
			EXPECT_EQ(ans.first, 1.0);
			EXPECT_LT(ans.second, 1e-5);
		}
		{
			auto ans = run(ti45);
			EXPECT_EQ(ans.first, 1.0);
			EXPECT_LT(ans.second, 1e-9);
		}
	};
}

flecsi::unit::driver<extest> driver;
}
