#include "flecsolve/util/config.hh"
#include "flecsolve/time-integrators/factory.hh"
#include "flecsolve/time-integrators/operator_adapter.hh"
#include "flecsolve/solvers/factory.hh"

#include "heat.hh"

namespace heat {

void time_integration(control_policy & cp) {
	flog(info) << "Time integration" << std::endl;

	auto & u = cp.u();
	auto & unew = cp.unew();

	using namespace flecsolve;
	using namespace flecsolve::time_integrator;

	auto [ti_settings, slv_settings] =
		read_config("implicit.cfg",
	                integrator_factory::options("time-integrator"),
	                krylov_factory::options("linear-solver"));
	auto F = op::make_shared<operator_adapter<heat_op>>(cp.diffusivity);
	auto ti = integrator_factory::make(
		ti_settings, u, F, krylov_factory::make_shared(slv_settings, u, F));

	auto output = [&]() {
		if (output_steps.value()) {
			std::string fname{"timestep" +
			                  std::to_string(ti.get_current_step())};
			flecsi::execute<task::output, flecsi::mpi>(
				flecsi::exec::on, cp.mesh(), u.data.ref(), fname.c_str());
		}
	};
	output();

	auto dt = ti.get_current_dt();
	while (ti.get_current_time() < ti.get_final_time()) {
		ti.advance(dt, u, unew);
		auto good_solution = ti.check_solution();
		if (good_solution) {
			flog(info) << "Step " << ti.get_current_step() << " advanced " << dt
					   << "s to time " << ti.get_current_time() << "s"
					   << std::endl;
			ti.update();
			std::swap(u, unew);
			output();
		}
		dt = ti.get_next_dt(good_solution);
	}
}

}
