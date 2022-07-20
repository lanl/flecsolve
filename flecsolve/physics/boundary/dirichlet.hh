#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <utility>
#include <vector>

#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/common/vector_types.hh"
#include "flecsolve/physics/tasks/operator_task.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve {
namespace physics {

template<auto Var,
         class Vec,
         topo_axes_t<Vec> Axis,
         topo_domain_t<Vec> Boundary>
struct dirichlet;

template<auto Var,
         class Vec,
         topo_axes_t<Vec> Axis,
         topo_domain_t<Vec> Boundary>
struct operator_parameters<dirichlet<Var, Vec, Axis, Boundary>> {
	static constexpr auto op_axis = Axis;
	static constexpr auto op_boundary = Boundary;
	scalar_t<Vec> boundary_value = 0.0;
};

namespace tasks {
template<auto Var,
         class Vec,
         topo_axes_t<Vec> Axis,
         topo_domain_t<Vec> Boundary>
struct operator_task<dirichlet<Var, Vec, Axis, Boundary>> {
	static constexpr void
	operate(topo_acc<Vec> m, field_acc<Vec, flecsi::ro> u, scalar_t<Vec> v) {
		auto jj = m.template get_stencil<Axis,
		                                 topo_t<Vec>::cells,
		                                 topo_t<Vec>::cells,
		                                 Boundary>(utils::offset_seq<>());

		for (auto j : jj) {
			u[j] = v;
		}
	}
};
} // tasks

template<auto Var,
         class Vec,
         topo_axes_t<Vec> Axis,
         topo_domain_t<Vec> Boundary>
struct dirichlet : operator_settings<dirichlet<Var, Vec, Axis, Boundary>> {

	using base_type = operator_settings<dirichlet<Var, Vec, Axis, Boundary>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;
	using task_type = typename base_type::task_type;

	dirichlet(param_type p) : base_type(p) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V &) const {
		const auto & subu = u.template subset(variable<Var>);
		flecsi::execute<task_type::operate>(
			subu.data.topo, subu.data.ref(), this->parameters.boundary_value);
	}
};

// template<class Topo, typename Topo::axis Axis, typename Topo::domain
// Boundary, class Scalar> dirchilet(Axis, Boundary, Scalar)->dirchilet<Topo,
// Axis, Boundary, Scalar>

}
}
