#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <flecsi/util/array_ref.hh>
#include <iterator>
#include <utility>
#include <vector>
#include <list>

#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/common/vector_types.hh"
#include "flecsolve/physics/tasks/operator_task.hh"
#include "flecsolve/vectors/data/mesh.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve {
namespace physics {

template<auto Var, class Vec>
struct coefficent;

template<auto Var, class Vec>
struct operator_parameters<coefficent<Var, Vec>> : components::FacesHandle<Vec> {
	using op_type = operator_parameters<coefficent<Var, Vec>>;
};

namespace tasks {
template<auto Var, class Vec>
struct operator_task<coefficent<Var, Vec>> {
	template<auto Axis>
	static void unit_coef(topo_acc<Vec> m,
	                                field_acc<Vec, ro> u,
	                                field_acc<Vec, wo> b) {
		auto jj = m.template get_stencil<Axis, topo_t<Vec>::faces>(
			utils::offset_seq<>());

		for (auto j : jj) {
			b[j] = 1.0;
		}
	}
};
}

template<auto Var, class Vec>
struct coefficent : operator_settings<coefficent<Var, Vec>> {

	using base_type = operator_settings<coefficent<Var, Vec>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;
	using task_type = typename base_type::task_type;

	coefficent(param_type p) : base_type(p) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V & v) const {
		const auto & subu = u.template subset(variable<Var>);
		const auto & subv = v.template subset(variable<Var>);
		sweep(subu.data.topo, subu.data.ref(), topo_axes_t<Vec>());
	}

	template<auto... Axis>
	constexpr void sweep(topo_slot_t<Vec> & m,
	                     cell_ref<Vec> u,
	                     flecsi::util::constants<Axis...>) const {

		(flecsi::execute<task_type::template unit_coef<Axis>>(
			 m, u, (*(this->parameters.faces))[Axis]),
		 ...);
	}
};
}
}