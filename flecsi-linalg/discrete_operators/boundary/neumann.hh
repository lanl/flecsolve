#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <utility>
#include <vector>

#include "flecsi-linalg/discrete_operators/common/operator_base.hh"
#include "flecsi-linalg/discrete_operators/tasks/operator_task.hh"

namespace flecsi {
namespace linalg {
namespace discrete_operators {

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar = double>
struct neumann;

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar>
struct operator_traits<neumann<Var, Topo, Axis, Boundary, Scalar>> {
	using scalar_t = Scalar;
	using topo_t = Topo;
	using topo_slot_t = flecsi::data::topology_slot<Topo>;
	using topo_axes_t = typename topo_t::axes;
	constexpr static auto dim = Topo::dimension;
	using tasks_f = tasks::topology_tasks<topo_t, field<scalar_t>>;

	using cell_ref =
		typename field<scalar_t>::template Reference<topo_t, topo_t::cells>;

	using face_ref =
		typename field<scalar_t>::template Reference<topo_t, topo_t::faces>;

	constexpr static auto op_axis = Axis;
	constexpr static auto op_boundary = Boundary;
};

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar>
struct operator_parameters<neumann<Var, Topo, Axis, Boundary, Scalar>> {
	using op_type = neumann<Var, Topo, Axis, Boundary, Scalar>;
	using face_ref = typename operator_traits<op_type>::face_ref;

	std::optional<face_ref> b;
};

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar>
struct neumann : operator_settings<neumann<Var, Topo, Axis, Boundary, Scalar>> {
	using base_type =
		operator_settings<neumann<Var, Topo, Axis, Boundary, Scalar>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;
	using topo_slot_t = typename operator_traits<exact_type>::topo_slot_t;
	using cell_ref = typename operator_traits<exact_type>::cell_ref;
	using tasks_f = typename operator_traits<exact_type>::tasks_f;

	neumann(param_type p) : base_type(p) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V & v) const {

		const auto & subu = u.template subset(variable<Var>);
		_apply(subu.data.topo, subu.data.ref());
	}

	// TODO: allow default value (=1.0)
	void _apply(topo_slot_t & m, cell_ref u) const {
		flecsi::execute<tasks_f::template boundary_fluxset<Axis, Boundary>>(
			m, *(this->parameters.b), u);
	}
};

} // namespace discrete_operators
} // namespace linalg
} // namespace flecsi
