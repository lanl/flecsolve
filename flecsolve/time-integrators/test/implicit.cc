#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/util/config.hh"
#include "flecsolve/time-integrators/bdf.hh"

#include "flecsolve/util/test/mesh.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/operators/core.hh"
#include <optional>

namespace flecsolve {

testmesh::slot msh;
testmesh::cslot coloring;

const flecsi::field<double>::definition<testmesh, testmesh::cells> xd, xnewd;

struct rate : op::base<> {
	rate(double lambda) : lambda(lambda), gamma(1.) {}

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
		y.axpy(-gamma, y, x);
	}

	template<class D, class R>
	void apply_rhs(const D & x, R & y) const {
		y.scale(lambda, x);
	}

	template<class V>
	bool is_valid(const V &) {
		return true;
	}

	double get_scaling() const { return gamma; }
	void set_scaling(double scaling) { gamma = scaling; }

	double get_rate() const { return lambda; }

protected:
	double lambda;
	double gamma;
};

struct rate_solver {
	template<class D, class R>
	solve_info apply(const D & b, R & x) const {
		auto rhs = b.min().get();
		const auto & op = F.source();
		auto sol = rhs / (1. - op.get_rate() * op.get_scaling());
		x.set_scalar(sol);

		solve_info info;
		info.status = solve_info::stop_reason::converged_atol;
		return info;
	}

	op::core<rate, op::shared_storage> F;
};

int bdftest() {
	using namespace flecsolve::time_integrator;

	UNIT () {
		double ic = 3.;

		init_mesh(1, msh, coloring);

		op::core<rate, op::shared_storage> F(-1.);
		auto x = vec::make(msh)(xd);
		auto xnew = vec::make(msh)(xnewd);

		rate_solver solver{F};
		bdf::parameters params2(
			"bdf-2", F, bdf::topo_work<>::get(x), std::ref(solver)),
			params5("bdf-5", F, bdf::topo_work<>::get(x), std::ref(solver));
		read_config("implicit.cfg", params2, params5);
		bdf::integrator ti2(std::move(params2));
		bdf::integrator ti5(std::move(params5));

		auto run = [&](auto & ti) {
			x.set_scalar(ic);
			auto dt = ti.get_current_dt();
			bool first_step = true;
			std::cout.precision(10);
			while (ti.get_current_time() < ti.get_final_time()) {
				ti.advance(dt, first_step, x, xnew);
				auto good_solution = ti.check_solution();
				if (good_solution) {
					ti.update();
					std::swap(x, xnew);
					first_step = false;
				}
				dt = ti.get_next_dt(good_solution);
			}

			auto sol =
				ic * std::exp(F.source().get_rate() * ti.get_final_time());
			auto approx = x.max().get();
			return std::tuple(ti.get_final_time(),
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
	};
}

flecsi::util::unit::driver<bdftest> driver;
}
