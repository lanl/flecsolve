#ifndef FLECSOLVE_EXAMPLES_HEAT_HEAT_H
#define FLECSOLVE_EXAMPLES_HEAT_HEAT_H

#include "flecsolve/operators/base.hh"

#include "control.hh"
#include "mesh.hh"

namespace heat {

inline flecsi::program_option<std::size_t>
	x_extents("x-extents", "The x extents of the mesh.", 1);
inline flecsi::program_option<std::size_t>
	y_extents("y-extents", "The y extents of the mesh.", 1);

inline flecsi::program_option<double>
	diffusivity("diffusivity",
                "diffusivity,d",
                "Diffusivity constant",
                {{flecsi::option_default, 1.}});

inline flecsi::program_option<bool>
	output_steps("output-steps",
                 "output,-o",
                 "Output all timesteps",
                 {{flecsi::option_default, false}});

void time_integration(control_policy &);

namespace task {

inline void output(mesh::accessor<ro> m,
                   field<double>::accessor<ro, ro> ua,
                   const char * base_fname) {
	auto u = m.mdcolex<mesh::vertices>(ua);

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

inline void laplace(mesh::accessor<ro> m,
                    const double c,
                    field<double>::accessor<wo, na> unewa,
                    field<double>::accessor<rw, ro> ua) {
	auto unew = m.mdcolex<mesh::vertices>(unewa);
	auto u = m.mdcolex<mesh::vertices>(ua);

	const auto dx_over_dy = m.xdelta() / m.ydelta();
	const auto dy_over_dx = m.ydelta() / m.xdelta();
	const auto dxdy = m.dxdy();

	// boundary conditions (Dirichlet)
	auto xverts = m.vertices<mesh::x_axis, mesh::all>();
	const auto is = *xverts.begin();
	const auto ie = *(xverts.end() - 1);
	if (m.is_boundary<mesh::x_axis, mesh::boundary::low>(is)) {
		for (auto j : m.vertices<mesh::y_axis, mesh::logical>()) {
			u(is, j) = 0.;
		}
	}

	if (m.is_boundary<mesh::x_axis, mesh::boundary::high>(ie)) {
		for (auto j : m.vertices<mesh::y_axis, mesh::logical>()) {
			u(ie, j) = 0.;
		}
	}

	auto yverts = m.vertices<mesh::y_axis, mesh::all>();
	const auto js = *yverts.begin();
	const auto je = *(yverts.end() - 1);
	if (m.is_boundary<mesh::y_axis, mesh::boundary::low>(js)) {
		for (auto i : m.vertices<mesh::x_axis, mesh::logical>()) {
			u(i, js) = 0.;
		}
	}

	if (m.is_boundary<mesh::y_axis, mesh::boundary::high>(je)) {
		for (auto i : m.vertices<mesh::x_axis, mesh::logical>()) {
			u(i, je) = 0.;
		}
	}

	for (auto j : m.vertices<mesh::y_axis>()) {
		for (auto i : m.vertices<mesh::x_axis>()) {
			unew(i, j) =
				c * (1. / dxdy) *
				(dy_over_dx * (u(i + 1, j) - 2 * u(i, j) + u(i - 1, j)) +
			     dx_over_dy * (u(i, j + 1) - 2 * u(i, j) + u(i, j - 1)));
		}
	}
}

}

struct heat_op : flecsolve::op::base<heat_op> {

	heat_op(double d) : diffusivity(d) {}

	template<class Domain, class Range>
	void apply(const Domain & x, Range & y) const {
		flecsi::execute<task::laplace>(
			y.data.topo(), diffusivity, y.data.ref(), x.data.ref());
	}

	double diffusivity;
};

}

#endif
