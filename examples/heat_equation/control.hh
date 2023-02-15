#ifndef FLECSOLVE_EXAMPLES_HEAT_CONTROL_H
#define FLECSOLVE_EXAMPLES_HEAT_CONTROL_H

#include <utility>

#include "flecsi/run/control.hh"

#include "flecsolve/vectors/mesh.hh"

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
	struct node_policy {};

	using control = flecsi::run::control<control_policy>;

	using control_points =
		list<point<cp::initialize>, point<cp::advance>, point<cp::finalize>>;

	double diffusivity;

	using vec = decltype(flecsolve::vec::mesh(m, ud[0](m)));
	vec & u() { return u_.value(); }

	vec & unew() { return unew_.value(); }

	void initialize_vectors() {
		u_.emplace(m, ud[0](m));
		unew_.emplace(m, ud[1](m));
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
