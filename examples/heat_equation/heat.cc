#include "flecsi/execution.hh"
#include "flecsi/flog.hh"

#include "heat.hh"

namespace heat {

namespace task {

void ics(mesh::accessor<ro> m, field<double>::accessor<wo, na> ua) {
	auto f = [](double x, double y) -> double {
		if (x >= 4 && x <= 6 && y >= 4 && y <= 6)
			return 50;
		else
			return 0.;
	};

	auto u = m.mdcolex<mesh::vertices>(ua);

	for (auto j : m.vertices<mesh::y_axis, mesh::logical>()) {
		const double y = m.value<mesh::y_axis>(j);
		for (auto i : m.vertices<mesh::x_axis, mesh::logical>()) {
			const double x = m.value<mesh::x_axis>(i);
			u(i, j) = f(x, y);
		}
	}
}

}

void init_mesh(control_policy & cp) {
	flog(info) << "Initializing " << x_extents.value() << "x"
			   << y_extents.value() << " mesh" << std::endl;
	flecsi::flog::flush();

	std::vector<std::size_t> axis_extents{x_extents.value(), y_extents.value()};

	coloring.allocate(flecsi::processes(), axis_extents);

	mesh::grect geometry;
	geometry[0][0] = 0.0;
	geometry[0][1] = 10.0;
	geometry[1] = geometry[0];

	m.allocate(coloring.get(), geometry);

	cp.diffusivity = diffusivity.value();
	cp.initialize_vectors();
	cp.save_geometry(geometry, axis_extents);
}
inline control::action<init_mesh, cp::initialize> init_mesh_action;

void initial_conditions(control_policy & cp) {
	flog(info) << "Setting intial conditions" << std::endl;
	flecsi::flog::flush();

	auto & u = cp.u();

	flecsi::execute<task::ics>(m, u.data.ref());
}
inline control::action<initial_conditions, cp::initialize> ic_action;
inline auto const dep = ic_action.add(init_mesh_action);

inline control::action<time_integration, cp::advance> ti_action;

void output_solution(control_policy & cp) {
	flecsi::execute<task::output, flecsi::mpi>(
		m, cp.u().data.ref(), "solution");
}
inline control::action<output_solution, cp::finalize> output_action;

}

int main(int argc, char * argv[]) {

	auto status = flecsi::initialize(argc, argv);
	status = heat::control::check_status(status);

	if (status != flecsi::run::status::success) {
		return status < flecsi::run::status::clean ? 0 : status;
	}

	flecsi::flog::add_output_stream("clog", std::clog, true);

	status = flecsi::start(heat::control::execute);

	flecsi::finalize();

	return 0;
}
