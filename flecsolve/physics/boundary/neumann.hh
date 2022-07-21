#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <utility>
#include <vector>

#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/tasks/operator_task.hh"

namespace flecsolve {
namespace physics {

template<class Vec,
         auto Axis,
         auto Boundary,
         auto Var = Vec::var.value>
struct neumann;

template<class Vec,
         auto Axis,
         auto Boundary,
         auto Var>
struct operator_parameters<neumann<Vec, Axis, Boundary, Var>> {
	using op_type = neumann<Vec, Axis, Boundary, Var>;
	static constexpr auto op_axis = Axis;
	static constexpr auto op_boundary = Boundary;
	scalar_t<Vec> val = 0.0;
};

namespace tasks {
template<class Vec,
         auto Axis,
         auto Boundary,
         auto Var>
struct operator_task<neumann<Vec, Axis, Boundary, Var>> {
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

template<class Vec,
         auto Axis,
         auto Boundary,
         auto Var>
struct neumann : operator_settings<neumann<Vec, Axis, Boundary, Var>> {
	using base_type = operator_settings<neumann<Vec, Axis, Boundary, Var>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;
	using task_type = typename base_type::task_type;

	neumann(param_type p) : base_type(p) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V &) const {
		const auto & subu = u.template subset(variable<Var>);
		flecsi::execute<task_type::operate>(
			subu.data.topo, subu.data.ref(), this->parameters.val);
	}
};

}
} // namespace physics
