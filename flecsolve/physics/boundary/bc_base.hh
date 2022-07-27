#pragma once

#include <array>
#include <utility>
#include <vector>

#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/common/vector_types.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve {
namespace physics {

template<class Spec, auto Axis, auto Boundary>
struct bc;

template<class Spec, auto Axis, auto Boundary>
struct operator_parameters<bc<Spec, Axis, Boundary>>
	: operator_parameters<Spec> {
	static constexpr auto op_axis = Axis;
	static constexpr auto op_boundary = Boundary;
};

template<class Spec, auto Axis, auto Boundary>
struct operator_traits<bc<Spec, Axis, Boundary>>
{
	static constexpr std::string_view label{"boundary_condition"};
};

template<class Spec, auto Axis, auto Boundary>
struct bc : operator_settings<bc<Spec, Axis, Boundary>> {

	using base_type = operator_settings<bc<Spec, Axis, Boundary>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;
	using task_type = typename base_type::task_type;


	bc(param_type p) : base_type(p) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V &) const {
		task_type::launch(u, this->parameters);
	}

};

}
}