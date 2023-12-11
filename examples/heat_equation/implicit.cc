#include "flecsolve/util/config.hh"
#include "flecsolve/time-integrators/bdf.hh"
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
	bdf::parameters params(
		"time-integrator",
		op::core<operator_adapter<heat_op>, op::value_storage>(cp.diffusivity),
		bdf::topo_work<>::get(u),
		krylov_factory());
	read_config("implicit.cfg", params);
	bdf::integrator ti(std::move(params));

	auto output = [&]() {
		if (output_steps.value()) {
			std::string fname{"timestep" +
			                  std::to_string(ti.get_current_step())};
			flecsi::execute<task::output, flecsi::mpi>(
				m, u.data.ref(), fname.c_str());
		}
	};
	output();

	auto dt = ti.get_current_dt();
	bool first_step = true;
	while (ti.get_current_time() < ti.get_final_time()) {
		ti.advance(dt, first_step, u, unew);
		auto good_solution = ti.check_solution();
		if (good_solution) {
			flog(info) << "Step " << ti.get_current_step() << " advanced " << dt
					   << "s to time " << ti.get_current_time() << "s"
					   << std::endl;
			ti.update();
			std::swap(u, unew);
			first_step = false;
			output();
		}
		dt = ti.get_next_dt(good_solution);
	}
}

}
