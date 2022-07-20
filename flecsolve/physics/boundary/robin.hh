#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <utility>
#include <vector>

#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/tasks/operator_task.hh"

namespace flecsolve {
namespace physics {

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar = double>
struct robin;

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar>
struct operator_traits<robin<Var, Topo, Axis, Boundary, Scalar>> {
	using scalar_t = Scalar;
	using topo_t = Topo;
	using topo_slot_t = flecsi::data::topology_slot<Topo>;
	using topo_axes_t = typename topo_t::axes;
	constexpr static auto dim = Topo::dimension;
	using tasks_f = tasks::topology_tasks<topo_t, flecsi::field<scalar_t>>;

	using cell_ref =
		typename flecsi::field<scalar_t>::template Reference<topo_t,
	                                                         topo_t::cells>;

	using face_ref =
		typename flecsi::field<scalar_t>::template Reference<topo_t,
	                                                         topo_t::faces>;

	constexpr static auto op_axis = Axis;
	constexpr static auto op_boundary = Boundary;
};

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar>
struct operator_parameters<robin<Var, Topo, Axis, Boundary, Scalar>> {
	using op_type = robin<Var, Topo, Axis, Boundary, Scalar>;
	using cell_ref = typename operator_traits<op_type>::cell_ref;
	using face_ref = typename operator_traits<op_type>::face_ref;

	Scalar alpha = 1.0;
	Scalar beta = 0.0;

	std::optional<face_ref> b;
};

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar>
struct robin : operator_settings<robin<Var, Topo, Axis, Boundary, Scalar>> {
	using base_type =
		operator_settings<robin<Var, Topo, Axis, Boundary, Scalar>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;
	using topo_slot_t = typename operator_traits<exact_type>::topo_slot_t;
	using cell_ref = typename operator_traits<exact_type>::cell_ref;
	using tasks_f = typename operator_traits<exact_type>::tasks_f;

	neumann(param_type p) : base_type(p) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V &) const {

		const auto & subu = u.template subset(variable<Var>);
		_apply(subu.data.topo, subu.data.ref());
	}

	void _apply(topo_slot_t & m, cell_ref u) const {
		if (this->parameters.b) {
			flecsi::execute<tasks_f::template boundary_robin<Axis, Boundary>>(
				m,
				u,
				*(this->parameters.b),
				this->parameters.alpha,
				this->parameters.beta);
		}
		else {
			flecsi::execute<tasks_f::template boundary_robin_1<Axis, Boundary>>(
				m, u, this->parameters.alpha, this->parameters.beta);
		}
	}
};

}
} // namespace physics
