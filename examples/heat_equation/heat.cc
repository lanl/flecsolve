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

	for (auto j : m.vertices<mesh::y_axis, mesh::extended>()) {
		const double y = m.value<mesh::y_axis>(j);
		for (auto i : m.vertices<mesh::x_axis, mesh::extended>()) {
			const double x = m.value<mesh::x_axis>(i);
			u(i, j) = f(x, y);
		}
	}
}

}

void init_mesh(control_policy & cp) {
	auto & sc = cp.scheduler();
	flog(info) << "Initializing " << x_extents.value() << "x"
			   << y_extents.value() << " mesh" << std::endl;
	flecsi::flog::flush();

	mesh::base::gcoord axis_extents{x_extents.value(), y_extents.value()};
	mesh::index_definition idef;
	idef.axes = mesh::base::make_axes(
		sc.runtime().processes(),
		axis_extents);

	for (auto & a : idef.axes)
		a.hdepth = a.bdepth = 1;

	mesh::grect geometry;
	geometry[0][0] = 0.0;
	geometry[0][1] = 10.0;
	geometry[1] = geometry[0];

	sc.allocate(cp.m, mesh::mpi_coloring(sc, idef), geometry);

	cp.diffusivity = diffusivity.value();
	cp.initialize_vectors();
	cp.save_geometry(geometry, axis_extents);
}
inline control::action<init_mesh, cp::initialize> init_mesh_action;

void initial_conditions(control_policy & cp) {
	flog(info) << "Setting intial conditions" << std::endl;
	flecsi::flog::flush();

	auto & u = cp.u();

	flecsi::execute<task::ics>(cp.mesh(), u.data.ref());
}
inline control::action<initial_conditions, cp::initialize> ic_action;
inline auto const dep = ic_action.add(init_mesh_action);

inline control::action<time_integration, cp::advance> ti_action;

void output_solution(control_policy & cp) {
	flecsi::execute<task::output, flecsi::mpi>(
		flecsi::exec::on, cp.mesh(), cp.u().data.ref(), "solution");
}
inline control::action<output_solution, cp::finalize> output_action;

}

int main(int argc, char * argv[]) {
	flecsi::getopt()(argc, argv);

	const flecsi::run::dependencies_guard dg;
	flecsi::run::config cfg;

	flecsi::runtime run(cfg);

	flecsi::flog::add_output_stream("clog", std::clog, true);

	return run.control<heat::control>();
}
