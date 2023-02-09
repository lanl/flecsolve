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

template<class Vec, auto Var>
struct operator_traits<neumann<Vec, Var>> {
	using op_type = neumann<Vec, Var>;
	static constexpr std::string_view label{"neumann"};
};

namespace tasks {
template<class Vec, auto Axis, auto Boundary, auto Var>
struct operator_task<bc<neumann<Vec, Var>, Axis, Boundary>> {
	template<class U, class P>
	static constexpr void launch(const U & u, P & p) {
		const auto & subu = u.template subset(variable<Var>);
		flecsi::execute<operate>(
			subu.data.topo(), subu.data.ref(), p.flux_value);
	}
	static constexpr void
	operate(topo_acc<Vec> m, field_acc<Vec, flecsi::rw> u, scalar_t<Vec> v) {
		constexpr int nd = (Boundary == topo_t<Vec>::domain::boundary_low ? 1 : -1);
		const scalar_t<Vec> dx = m.template dx<Axis>();
		const scalar_t<Vec> d_shift = static_cast<scalar_t<Vec>>(nd) * dx * v;

		auto uv = m.template mdspan<topo_t<Vec>::cells>(u);
		fvmtools::apply_to_with_index(
			uv,
			m.template full_range<topo_t<Vec>::cells, Axis, Boundary>(),
			[&](const auto k, const auto j, const auto i, auto xv) {
				if constexpr (Axis == topo_t<Vec>::x_axis)
					return xv[k][j][i + nd] - d_shift;
				else if constexpr (Axis == topo_t<Vec>::y_axis)
					return xv[k][j + nd][i] - d_shift;
				else if constexpr (Axis == topo_t<Vec>::z_axis)
					return xv[k + nd][j][i] - d_shift;
			},
			uv);
	}
};
} // tasks

}
} // namespace physics
