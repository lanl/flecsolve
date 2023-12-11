#include "flecsi/execution.hh"
#include "flecsi/flog.hh"

#include "flecsolve/solvers/factory.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/krylov_operator.hh"

#include "control.hh"
#include "mesh.hh"
#include "poisson.hh"

namespace poisson {

inline flecsi::program_option<std::size_t>
	x_extents("x-extents", "The x extents of the mesh.", 1);
inline flecsi::program_option<std::size_t>
	y_extents("y-extents", "The y extents of the mesh.", 1);
inline flecsi::program_option<bool>
	output_solution("output-solution",
                    "output,-o",
                    "Output solution",
                    {{flecsi::option_default, false}});

namespace task {

constexpr double pi = M_PI;

void set_problem(mesh::accessor<ro> m,
                 stencil_field<five_pt>::accessor<wo, na> soa,
                 field<double>::accessor<wo, na> fa,
                 field<double>::accessor<wo, na> ua) {

	auto rhs = [](double x, double y) {
		return 8 * (pi * pi) * sin(2 * pi * x) * sin(2 * pi * y);
	};

	auto sol = [](double x, double y) {
		return sin(2 * pi * x) * sin(2 * pi * y);
	};

	auto u = m.mdcolex<mesh::vertices>(ua);
	auto f = m.mdcolex<mesh::vertices>(fa);
	auto so = m.stencil_op<mesh::vertices, five_pt>(soa);

	const auto hx = m.xdelta();
	const auto hy = m.ydelta();
	const auto xh = hy / hx;
	const auto yh = hx / hy;
	const auto h2 = m.dxdy();

	std::size_t ibeg{2}, jbeg{2};
	if (m.is_boundary<mesh::x_axis, mesh::boundary::low>(ibeg - 1))
		++ibeg;
	if (m.is_boundary<mesh::y_axis, mesh::boundary::low>(jbeg - 1))
		++jbeg;

	auto ext = std::array{m.size<mesh::axis::x_axis, mesh::domain::all>(),
	                      m.size<mesh::axis::y_axis, mesh::domain::all>()};
	std::size_t iend{ext[0] - 2}, jend{ext[1] - 2};
	if (m.is_boundary<mesh::x_axis, mesh::boundary::high>(iend))
		--iend;
	if (m.is_boundary<mesh::y_axis, mesh::boundary::high>(jend))
		--jend;

	for (std::size_t j{2}; j <= jend; ++j) {
		for (std::size_t i{ibeg}; i <= iend; ++i) {
			so(i, j, five_pt::w) = xh;
		}
	}

	for (std::size_t j{jbeg}; j <= jend; ++j) {
		for (std::size_t i{2}; i <= iend; ++i) {
			so(i, j, five_pt::s) = yh;
		}
	}

	for (auto j : m.vertices<mesh::y_axis>()) {
		const double y = m.value<mesh::y_axis>(j);
		for (auto i : m.vertices<mesh::x_axis>()) {
			const double x = m.value<mesh::x_axis>(i);
			f(i, j) = rhs(x, y) * h2;
			u(i, j) = sol(x, y);

			so(i, j, five_pt::c) = 2 * xh + 2 * yh;
		}
	}
}

double scale(mesh::accessor<ro> m, double nrm) { return m.dxdy() * nrm; }

void output(mesh::accessor<ro> m,
            field<double>::accessor<ro, na> xa,
            const char * base_fname) {
	auto u = m.mdcolex<mesh::vertices>(xa);

	std::ofstream ofile(std::string{base_fname} + "-" +
	                    std::to_string(flecsi::process()) + ".dat");

	for (auto j : m.vertices<mesh::y_axis, mesh::logical>()) {
		const double y = m.value<mesh::y_axis>(j);
		for (auto i : m.vertices<mesh::x_axis, mesh::logical>()) {
			const double x = m.value<mesh::x_axis>(i);
			ofile << x << " " << y << " " << u(i, j) << '\n';
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
	geometry[0][1] = 1.0;
	geometry[1] = geometry[0];

	m.allocate(coloring.get(), geometry);

	cp.initialize_vectors();
}
inline control::action<init_mesh, cp::initialize> init_mesh_action;

void init_problem(control_policy & cp) {
	flecsi::execute<task::set_problem>(
		m, sod(m), cp.f().data.ref(), cp.sol().data.ref());
}
inline control::action<init_problem, cp::initialize> init_problem_action;

void solve(control_policy & cp) {
	using namespace flecsolve;

	auto & f = cp.f();
	auto & u = cp.u();

	std::size_t iter{0};
	op::krylov_parameters params(cg::settings("solver"),
	                             cg::topo_work<>::get(f),
	                             op::make(poisson_op{sod(m)}),
	                             op::I,
	                             [&](const auto &, double rnorm) {
									 std::cout << ++iter << " " << rnorm
											   << std::endl;
									 return false;
								 });
	read_config("poisson.cfg", params);

	op::krylov slv(std::move(params));

	slv.apply(f, u);
}
inline control::action<solve, cp::solve> solve_action;

void output(control_policy & cp) {
	if (output_solution.value()) {
		flecsi::execute<task::output, flecsi::mpi>(
			m, cp.u().data.ref(), "solution");
	}
}
inline control::action<output, cp::finalize> output_action;

void check_error(control_policy & cp) {
	auto & u = cp.u();
	auto & sol = cp.sol();
	u.subtract(u, sol);
	auto nrm = u.l2norm().get();
	auto err = flecsi::execute<task::scale>(m, nrm * nrm).get();
	flog(info) << "Error: " << err << std::endl;
}
inline control::action<check_error, cp::finalize> err_action;

}

int main(int argc, char * argv[]) {

	auto status = flecsi::initialize(argc, argv);
	status = poisson::control::check_status(status);

	if (status != flecsi::run::status::success) {
		return status < flecsi::run::status::clean ? 0 : status;
	}

	flecsi::flog::add_output_stream("clog", std::clog, true);

	status = flecsi::start(poisson::control::execute);

	flecsi::finalize();

	return 0;
}
