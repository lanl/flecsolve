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

const flecsi::field<double>::definition<testmesh, testmesh::cells> xd, xnewd;

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

		vec::mesh x(msh, xd(msh)), xnew(msh, xnewd(msh));

		rk23::parameters params23_var(
			"variable", std::ref(F), rk23::topo_work<>::get(x)),
			params23_fixed("fixed", std::ref(F), rk23::topo_work<>::get(x));
		rk45::parameters params45_var(
			"variable", std::ref(F), rk45::topo_work<>::get(x)),
			params45_fixed("fixed", std::ref(F), rk45::topo_work<>::get(x));
		read_config("explicit.cfg", params23_var, params23_fixed);
		read_config("explicit.cfg", params45_var, params45_fixed);
		rk23::integrator ti23_var(std::move(params23_var)),
			ti23_fixed(std::move(params23_fixed));
		rk45::integrator ti45_var(std::move(params45_var)),
			ti45_fixed(std::move(params45_fixed));

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

			auto sol = ic * std::exp(F.lambda * ti.get_final_time());
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
