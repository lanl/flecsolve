/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
#ifndef FLECSOLVE_EXAMPLES_HEAT_CONTROL_H
#define FLECSOLVE_EXAMPLES_HEAT_CONTROL_H

#include <utility>

#include "flecsi/run/control.hh"

#include "flecsolve/vectors/topo_view.hh"

#include "mesh.hh"

namespace heat {
enum class cp { initialize, advance, finalize };
inline const char * operator*(cp control_point) {
	switch (control_point) {
		case cp::initialize:
			return "initialize";
		case cp::advance:
			return "advance";
		default: // case cp::finalize:
			return "finalize";
	}
}
struct control_policy : flecsi::run::control_base {
	using control_points_enum = cp;

	using control = flecsi::run::control<control_policy>;

	using control_points =
		list<point<cp::initialize>, point<cp::advance>, point<cp::finalize>>;

	double diffusivity;
	heat::mesh::ptr m;

	auto & mesh() { return *m; }

	using vec = decltype(flecsolve::vec::make(ud[0](*m)));
	vec & u() { return u_.value(); }

	vec & unew() { return unew_.value(); }

	void initialize_vectors() {
		u_.emplace(flecsolve::vec::make(ud[0](mesh())));
		unew_.emplace(flecsolve::vec::make(ud[1](mesh())));
	}

	template<class T>
	void save_geometry(const T & g, std::vector<std::size_t> extents) {
		dx = std::abs(g[0][1] - g[0][0]) / (extents[0] - 1);
		dy = std::abs(g[1][1] - g[1][0]) / (extents[1] - 1);
	}

	double dx, dy;

protected:
	std::optional<vec> u_;
	std::optional<vec> unew_;
};

using control = flecsi::run::control<control_policy>;
}

#endif
