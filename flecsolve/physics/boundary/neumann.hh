#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <utility>
#include <vector>

#include "flecsolve/physics/common/vector_types.hh"
#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/boundary/bc_base.hh"

namespace flecsolve {
namespace physics {

template<class Vec, auto Var = Vec::var.value>
struct neumann;

template<class Vec, auto Var>
struct operator_parameters<neumann<Vec, Var>> {
	scalar_t<Vec> flux_value = 0.0;
};

namespace tasks {
template<class Vec, auto Axis, auto Boundary, auto Var>
struct operator_task<bc<neumann<Vec, Var>, Axis, Boundary>> {
	template<class U, class P>
	static constexpr void launch(const U & u, P & p) {
		const auto & subu = u.template subset(variable<Var>);
		flecsi::execute<operate>(subu.data.topo, subu.data.ref(), p.flux_value);
	}
	static constexpr void
	operate(topo_acc<Vec> m, field_acc<Vec, flecsi::rw> u, scalar_t<Vec> v) {
		const scalar_t<Vec> dx = m.template dx<Axis>();
		constexpr int nd = (Boundary == topo_t<Vec>::boundary_low ? 1 : -1);
		auto [jj, jo] =
			m.template get_stencil<Axis,
		                           topo_t<Vec>::cells,
		                           topo_t<Vec>::cells,
		                           Boundary>(utils::offset_seq<nd>());

		for (auto j : jj) {
			u[j] = u[j + jo] - static_cast<scalar_t<Vec>>(nd) * dx * v;
		}
	}
};
} // tasks

}
} // namespace physics
