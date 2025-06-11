#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/util/config.hh"
#include "flecsolve/time-integrators/bdf.hh"

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
	rate_solver(op::handle<op::core<rate>> oph) : F(oph) {}

	template<class D, class R>
	solve_info apply(const D & b, R & x) const {
		auto rhs = b.min().get();
		const auto & op = F.get();
		auto sol = rhs / (1. - op.get_rate() * op.get_scaling());
		x.set_scalar(sol);

		solve_info info;
		info.status = solve_info::stop_reason::converged_atol;
		return info;
	}

	op::handle<op::core<rate>> F;
};

int bdftest() {
	using namespace flecsolve::time_integrator;

	UNIT () {
		testmesh::slot msh;

		double ic = 3.;

		init_mesh(1, msh);

		auto F = op::make_shared<rate>(-1.);
		auto x = vec::make(msh)(xd);
		auto xnew = vec::make(msh)(xnewd);

		auto solver = op::make_shared<rate_solver>(F);
		auto [ti2, ti5] = std::apply(
			[&](auto &&... s) {
				return std::make_tuple(bdf::integrator(bdf::parameters(
					                                       s, F, bdf::make_work(x), solver))...);
			},
			read_config(
				"implicit.cfg", bdf::options("bdf-2"), bdf::options("bdf-5")));

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
				ic * std::exp(get_rate(F) * ti.get_final_time());
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
