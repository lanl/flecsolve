#include "flecsolve/util/config.hh"
#include "flecsolve/time-integrators/rk23.hh"

#include "heat.hh"

namespace heat {

void time_integration(control_policy & cp) {
	flog(info) << "Time integration" << std::endl;
	flecsi::flog::flush();

	auto & u = cp.u();
	auto & unew = cp.unew();

	using namespace flecsolve::time_integrator;

	rk23::parameters params(
		"time-integrator", heat_op{cp.diffusivity}, rk23::topo_work<>::get(u));
	flecsolve::read_config("explicit.cfg", params);

	rk23::integrator ti(std::move(params));

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
	while (ti.get_current_time() < ti.get_final_time()) {
		ti.advance(dt, u, unew);
		auto good_solution = ti.check_solution();
		if (good_solution) {
			flog(info) << "Step " << ti.get_current_step() << " advanced " << dt
					   << "s to time " << ti.get_current_time() << "s"
					   << std::endl;
			ti.update();
			output();
			std::swap(u, unew);
		}
		dt = ti.get_next_dt(good_solution);
	}
}

}
