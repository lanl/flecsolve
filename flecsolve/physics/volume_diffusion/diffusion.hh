#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <flecsi/util/array_ref.hh>
#include <flecsi/util/constant.hh>
#include <tuple>
#include <utility>

#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/common/operator_utils.hh"
#include "flecsolve/physics/expressions/operator_expression.hh"
#include "flecsolve/physics/common/vector_types.hh"
#include "flecsolve/physics/common/state_store.hh"
#include "flecsolve/physics/tasks/operator_task.hh"

namespace flecsolve {
namespace physics {

/**
 * @brief operator of finite-volume diffusion
 *
 * linear operators of the form -β ∇ (b ∇ u ) + α a u
 * where α and β  are constants, a is a cell-centered
 * array,b is a face-centered array, and u is a cell-centered
 * array.
 *
 * @tparam Var variable to apply on
 * @tparam Topo topology
 * @tparam Scalar scalar data-type
 */
template<class Vec, auto Var = Vec::var.value>
struct diffusion;

template<class Vec, auto Var>
struct operator_parameters<diffusion<Vec, Var>> {
	using op_type = diffusion<Vec, Var>;

	components::cells_handle<Vec> a;
	components::faces_handle<Vec> b;
	scalar_t<Vec> beta = 1.0;
	scalar_t<Vec> alpha = 0.0;
};

namespace tasks {
template<class Vec, auto Var>
struct operator_task<diffusion<Vec, Var>> {
	template<class U, class V, class Fluxes, class Du, class Par>
	static void launch(const U & u, V & v, Fluxes & fluxes, Du & du, Par & p) {
		auto & subu = u.template subset(variable<Var>);
		auto & subv = v.template subset(variable<Var>);

		auto & m = subu.data.topo;

		auto su = subu.data.ref();
		auto sv = subv.data.ref();

		flecsi::execute<zero>(m, sv);
		flecsi::execute<zero>(m, du);

		// determine the fluxes along the axis
		sweep(m, su, *(p.b), fluxes, du, topo_axes_t<Vec>());

		// collect all prior calculations and apply to range vector
		flecsi::execute<operate>(m, p.beta, p.alpha, *(p.a), su, du, sv);
	}

	template<auto Axis>
	static void update_flux(topo_acc<Vec> m,
	                        field_acc_all<Vec, ro> u_x,
	                        field_acc<Vec, ro> b_x,
	                        field_acc<Vec, wo> fu_x) {
		const scalar_t<Vec> dA = m.template normal_dA<Axis>();
		const scalar_t<Vec> idx = 1.0 / m.template dx<Axis>();

		auto [jj, jm1] =
			m.template get_stencil<Axis,
		                           topo_t<Vec>::cells,
		                           topo_t<Vec>::faces>(utils::offset_seq<-1>());

		for (auto j : jj) {
			fu_x[j] = b_x[j] * (dA * idx) * (u_x[j] - u_x[j + jm1]);
		}
	}

	template<auto Axis>
	static void sum_cell_flux(topo_acc<Vec> m,
	                          field_acc<Vec, ro> fu_x,
	                          field_acc<Vec, wo> du) {
		const scalar_t<Vec> i_dx = 1.0 / m.template dx<Axis>();
		auto [jj, jp1] = m.template get_stencil<Axis, topo_t<Vec>::cells>(
			utils::offset_seq<1>());

		for (auto j : jj) {
			du[j] += (fu_x[j + jp1] - fu_x[j]) * i_dx;
		}
	}

	static void operate(topo_acc<Vec> m,
	                    scalar_t<Vec> beta,
	                    scalar_t<Vec> alpha,
	                    field_acc_all<Vec, ro> a,
	                    field_acc_all<Vec, ro> u,
	                    field_acc_all<Vec, ro> du,
	                    field_acc_all<Vec, wo> un) {
		auto jj =
			m.template get_stencil<topo_t<Vec>::x_axis, topo_t<Vec>::cells>(
				utils::offset_seq<>());

		for (auto j : jj) {
			un[j] = (-beta * du[j]) + (alpha * a[j] * u[j]);
		}
	}

	static void zero(topo_acc<Vec> m, field_acc_all<Vec, wo> u) {
		auto jj =
			m.template get_stencil<topo_t<Vec>::x_axis,
		                           topo_t<Vec>::cells,
		                           topo_t<Vec>::cells,
		                           topo_t<Vec>::all>(utils::offset_seq<>());
		for (auto j : jj) {
			u[j] = 0.0;
		}
	}

	template<auto... Axis>
	static constexpr void sweep(topo_slot_t<Vec> & m,
	                            cell_ref<Vec> u,
	                            axes_set<face_ref<Vec>, Vec> b,
	                            std::array<face_ref<Vec>, sizeof...(Axis)> flx,
	                            cell_ref<Vec> du,
	                            flecsi::util::constants<Axis...>) {

		(flecsi::execute<update_flux<Axis>>(m, u, b[Axis], flx[Axis]), ...);

		(flecsi::execute<sum_cell_flux<Axis>>(m, flx[Axis], du), ...);
	}
};
}

template<class Vec, auto Var>
struct diffusion : operator_settings<diffusion<Vec, Var>> {

	using base_type = operator_settings<diffusion<Vec, Var>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;
	using task_type = typename base_type::task_type;

	static constexpr std::size_t dim = topo_axes_t<Vec>::size;

	// TODO: just use local memory
	using flux_store_t =
		common::topo_state_store<exact_type, Vec, topo_t<Vec>::faces, dim, 0>;
	// zero-D quantity change due to flux integration on surface
	using du_store_t =
		common::topo_state_store<exact_type, Vec, topo_t<Vec>::cells, 1, 0>;

	// These arrays hold references to the fields provided by the above
	// declarations
	std::array<face_ref<Vec>, dim> fluxes;

	cell_ref<Vec> du;

	diffusion(topo_slot_t<Vec> & s, param_type parameters)
		: base_type(parameters), fluxes(flux_store_t::get_state(s)),
		  du(du_store_t::get_state(s)) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V & v) const {

		task_type::launch(u, v, fluxes, du, this->parameters);
	}
};

template<class Derived>
struct operator_creator;

template<class Vec, auto Var>
struct operator_creator<diffusion<Vec, Var>> {
	template<template<class, auto> class CoeffOp,
	         class FacesRefArr,
	         class CellsRef>
	static constexpr decltype(auto) create(FacesRefArr fra,
	                                       CellsRef cr,
	                                       scalar_t<Vec> beta,
	                                       scalar_t<Vec> alpha,
	                                       topo_slot_t<Vec> & m) {
		auto coeffop = CoeffOp<Vec, Var>::create({fra});
		auto voldiff = diffusion<Vec, Var>::create({cr, fra, beta, alpha}, m);
		return op_expr(coeffop, voldiff);
	}
};

}
}
