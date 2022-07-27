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


template<class Vec, auto Var = Vec::var.value>
struct constant_coefficent;

template<class Vec, auto Var>
struct operator_parameters<constant_coefficent<Vec, Var>> {
	using op_type = operator_parameters<constant_coefficent<Vec, Var>>;
	components::faces_handle<Vec> faces;
	scalar_t<Vec> coeff_value = 1.0;
};

template<class Vec, auto Var>
struct operator_traits<constant_coefficent<Vec, Var>> {
	using op_type = constant_coefficent<Vec, Var>;
	static constexpr std::string_view label{"constant_coefficent"};
};

namespace tasks {
template<class Vec, auto Var>
struct operator_task<constant_coefficent<Vec, Var>> {
	template<class U, class Par>
	static void launch(const U & u, Par & p) {
		auto & subu = u.template subset(variable<Var>);
		sweep(subu.data.topo, *(p.faces), p.coeff_value, topo_axes_t<Vec>());
	}

	template<auto Axis>
	static void unit_coef(topo_acc<Vec> m,
	                                field_acc<Vec, flecsi::wo> b,scalar_t<Vec> c) {
		auto jj = m.template get_stencil<Axis, topo_t<Vec>::faces>(
			utils::offset_seq<>());

		for (auto j : jj) {
			b[j] = c;
		}
	}

	template<auto... Axis>
	static constexpr void sweep(topo_slot_t<Vec> & m,
						 axes_set<face_ref<Vec>, Vec> b,
						 scalar_t<Vec> val,
	                     flecsi::util::constants<Axis...>) {

		(flecsi::execute<unit_coef<Axis>>(
			 m, b[Axis], val ),
		 ...);
	}
};
}

template<class Vec, auto Var>
struct constant_coefficent : operator_settings<constant_coefficent<Vec, Var>> {

	using base_type = operator_settings<constant_coefficent<Vec, Var>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;
	using task_type = typename base_type::task_type;


	constant_coefficent(param_type p) : base_type(p) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V & v) const {
		const auto & subu = u.template subset(variable<Var>);
		const auto & subv = v.template subset(variable<Var>);
		// sweep(subu.data.topo, subu.data.ref(), topo_axes_t<Vec>());
		task_type::launch(u, this->parameters);
	}

	// template<auto... Axis>
	// constexpr void sweep(topo_slot_t<Vec> & m,
	//                      cell_ref<Vec> u,
	//                      flecsi::util::constants<Axis...>) const {

	// 	(flecsi::execute<task_type::template unit_coef<Axis>>(
	// 		 m, u, (*(this->parameters.faces))[Axis]),
	// 	 ...);
	// }
};
}
}