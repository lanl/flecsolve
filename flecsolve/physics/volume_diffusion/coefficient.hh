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
#include "flecsolve/physics/specializations/fvm_narray.hh"
#include "flecsolve/vectors/data/mesh.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve {
namespace physics {

template<class Spec, class Vec>
struct coefficient;

template<class Spec, class Vec>
struct operator_parameters<coefficient<Spec, Vec>> : operator_parameters<Spec> {
	components::faces_handle<Vec> faces;
};

template<class Spec, class Vec>
struct operator_traits<coefficient<Spec, Vec>> {
	static constexpr std::string_view label{"coefficient_setter"};
};

template<class Spec, class Vec>
struct coefficient : operator_settings<coefficient<Spec, Vec>> {

	using base_type = operator_settings<coefficient<Spec, Vec>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;
	using task_type = typename base_type::task_type;

	coefficient(param_type p) : base_type(p) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V &) const {
		task_type::launch(u, this->parameters);
	}
};

template<class Vec, auto Var = Vec::var.value>
struct constant_coefficient;

template<class Vec, auto Var>
struct operator_parameters<constant_coefficient<Vec, Var>> {
	using op_type = operator_parameters<constant_coefficient<Vec, Var>>;
	// components::faces_handle<Vec> faces;
	scalar_t<Vec> coeff_value = 1.0;
};

template<class Vec, auto Var>
struct operator_traits<constant_coefficient<Vec, Var>> {
	using op_type = constant_coefficient<Vec, Var>;
	static constexpr std::string_view label{"constant_coefficient"};
};

namespace tasks {
template<class Vec, auto Var>
struct operator_task<coefficient<constant_coefficient<Vec, Var>, Vec>> {
	template<class U, class Par>
	static void launch(const U & u, Par & p) {
		auto & subu = u.template subset(variable<Var>);
		flecsi::execute<coef>(subu.data.topo(),
		                      (*(p.faces))[topo_t<Vec>::x_axis],
		                      (*(p.faces))[topo_t<Vec>::y_axis],
		                      (*(p.faces))[topo_t<Vec>::z_axis],
		                      p.coeff_value);
	}

	static void coef(topo_acc<Vec> m,
	                 field_acc<Vec, flecsi::wo> b_x,
	                 field_acc<Vec, flecsi::wo> b_y,
	                 field_acc<Vec, flecsi::wo> b_z,
	                 scalar_t<Vec> c) {

		auto bvx = m.template mdspan<topo_t<Vec>::faces>(b_x);
		auto bvy = m.template mdspan<topo_t<Vec>::faces>(b_y);
		auto bvz = m.template mdspan<topo_t<Vec>::faces>(b_z);
		fvmtools::apply_to(
			bvx,
			m.template full_range<topo_t<Vec>::faces, topo_t<Vec>::x_axis>(),
			[&]() { return c; });
		fvmtools::apply_to(
			bvy,
			m.template full_range<topo_t<Vec>::faces, topo_t<Vec>::y_axis>(),
			[&]() { return c; });
		fvmtools::apply_to(
			bvz,
			m.template full_range<topo_t<Vec>::faces, topo_t<Vec>::z_axis>(),
			[&]() { return c; });
	}
};
}

template<class Vec, auto Var = Vec::var.value>
struct average_coefficient;

template<class Vec, auto Var>
struct operator_parameters<average_coefficient<Vec, Var>> {
	using op_type = operator_parameters<average_coefficient<Vec, Var>>;
	scalar_t<Vec> dummy_value = 1.0;
};

template<class Vec, auto Var>
struct operator_traits<average_coefficient<Vec, Var>> {
	using op_type = average_coefficient<Vec, Var>;
	static constexpr std::string_view label{"average_coefficient"};
};

namespace tasks {
template<class Vec, auto Var>
struct operator_task<coefficient<average_coefficient<Vec, Var>, Vec>> {
	template<class U, class Par>
	static void launch(const U & u, Par & p) {
		auto & subu = u.template subset(variable<Var>);
		flecsi::execute<coef>(subu.data.topo(),
		                      subu.data.ref(),
		                      (*(p.faces))[topo_t<Vec>::x_axis],
		                      (*(p.faces))[topo_t<Vec>::y_axis],
		                      (*(p.faces))[topo_t<Vec>::z_axis]);
	}

	static void coef(topo_acc<Vec> m,

	                 field_acc<Vec, flecsi::rw> u,
	                 field_acc<Vec, flecsi::wo> b_x,
	                 field_acc<Vec, flecsi::wo> b_y,
	                 field_acc<Vec, flecsi::wo> b_z) {

		auto uv = m.template mdspan<topo_t<Vec>::cells>(u);
		auto bvx = m.template mdspan<topo_t<Vec>::faces>(b_x);
		auto bvy = m.template mdspan<topo_t<Vec>::faces>(b_y);
		auto bvz = m.template mdspan<topo_t<Vec>::faces>(b_z);
		fvmtools::apply_to_with_index(
			bvx,
			m.template full_range<topo_t<Vec>::faces, topo_t<Vec>::x_axis>(),
			[&](const auto k, const auto j, const auto i) {
				return 0.5 * (uv[k][j][i] + uv[k][j][i - 1]);
			});
		fvmtools::apply_to_with_index(
			bvy,
			m.template full_range<topo_t<Vec>::faces, topo_t<Vec>::y_axis>(),
			[&](const auto k, const auto j, const auto i) {
				return 0.5 * (uv[k][j][i] + uv[k][j - 1][i]);
			});
		fvmtools::apply_to_with_index(
			bvz,
			m.template full_range<topo_t<Vec>::faces, topo_t<Vec>::z_axis>(),
			[&](const auto k, const auto j, const auto i) {
				return 0.5 * (uv[k][j][i] + uv[k - 1][j][i]);
			});
	}
};
}

}
}
