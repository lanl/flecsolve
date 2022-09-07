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
#include "flecsolve/physics/specializations/fvm_narray.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve {
namespace physics {

/// forward-declare operator
template<class Vec, auto Var = Vec::var.value>
struct diffusion;

/**
 * parameters of `diffusion` operator
 *
 * @tparam Vec flecsolve::vec::msh object
 * @tparam Var (optional) specify the variable if a multivector
 */
template<class Vec, auto Var>
struct operator_parameters<diffusion<Vec, Var>> {
	/// cell-centered components
	components::cells_handle<Vec> a;
	/// face-centered components (3-component)
	components::faces_handle<Vec> b;
	/// beta constant
	scalar_t<Vec> beta = 1.0;
	/// alpha constant
	scalar_t<Vec> alpha = 0.0;
};

template<class Vec, auto Var>
struct operator_traits<diffusion<Vec, Var>> {
	using op_type = diffusion<Vec, Var>;
	static constexpr std::string_view label{"diffusion"};
};

namespace tasks {
/**
 * task specialization for of `diffusion` operator
 *
 * @tparam Vec flecsolve::vec::msh object
 * @tparam Var (optional) specify the variable if a multivector
 */
template<class Vec, auto Var>
struct operator_task<diffusion<Vec, Var>> {
	/**
	 * calculates the flux at cell faces, and applies surface and volume
	 * operators variables following domain/range vectors are passed from the
	 * operator `diffusion` `apply()`
	 *
	 * @param u domain vector
	 * @param v range vector
	 * @param fluxes an axis-indexed (x,y,z) array of face-centered field
	 * references
	 * @param cell-centered storage that holds the value of the surface
	 * integration
	 * @param p the parameter struct of the operator
	 *
	 */
	template<class U, class V, class Fluxes, class Du, class Par>
	static void launch(const U & u, V & v, Fluxes & fluxes, Du & du, Par & p) {
		auto & subu = u.template subset(variable<Var>);
		auto & subv = v.template subset(variable<Var>);

		auto & m = subu.data.topo;

		auto su = subu.data.ref();
		auto sv = subv.data.ref();
		// we first calculated the x,y,z face-centered fluxes, using the
		// corrisponding face-centered diffusion coefficents
		flecsi::execute<update_flux>(m,
		                             su,
		                             (*(p.b))[topo_t<Vec>::x_axis],
		                             (*(p.b))[topo_t<Vec>::y_axis],
		                             (*(p.b))[topo_t<Vec>::z_axis],
		                             fluxes[0],
		                             fluxes[1],
		                             fluxes[2]);

		// the fluxes then are summed to realize the surface integration
		flecsi::execute<sum_cell_flux>(m, fluxes[0], fluxes[1], fluxes[2], du);

		// the surface integration is combined with the volume integration
		flecsi::execute<operate>(m, p.beta, p.alpha, (*(p.a)), su, du, sv);
	}

	// TODO: implement "looping" over axes
	static void update_flux(topo_acc<Vec> m,
	                        field_acc_all<Vec, flecsi::ro> u,
	                        field_acc<Vec, flecsi::ro> b_x,
	                        field_acc<Vec, flecsi::ro> b_y,
	                        field_acc<Vec, flecsi::ro> b_z,
	                        field_acc<Vec, flecsi::rw> fu_x,
	                        field_acc<Vec, flecsi::rw> fu_y,
	                        field_acc<Vec, flecsi::rw> fu_z) {

		const scalar_t<Vec> dAx = m.template normal_dA<topo_t<Vec>::x_axis>();
		const scalar_t<Vec> dAy = m.template normal_dA<topo_t<Vec>::y_axis>();
		const scalar_t<Vec> dAz = m.template normal_dA<topo_t<Vec>::z_axis>();

		const scalar_t<Vec> i_dx = 1.0 / m.template dx<topo_t<Vec>::x_axis>();
		const scalar_t<Vec> i_dy = 1.0 / m.template dx<topo_t<Vec>::y_axis>();
		const scalar_t<Vec> i_dz = 1.0 / m.template dx<topo_t<Vec>::z_axis>();

		auto uv = m.template mdspan<topo_t<Vec>::cells>(u);
		auto bvx = m.template mdspan<topo_t<Vec>::faces>(b_x);
		auto bvy = m.template mdspan<topo_t<Vec>::faces>(b_y);
		auto bvz = m.template mdspan<topo_t<Vec>::faces>(b_z);

		auto fvx = m.template mdspan<topo_t<Vec>::faces>(fu_x);
		auto fvy = m.template mdspan<topo_t<Vec>::faces>(fu_y);
		auto fvz = m.template mdspan<topo_t<Vec>::faces>(fu_y);

		fvmtools::apply_to_with_index(
			fvx,
			m.template full_range<topo_t<Vec>::faces, topo_t<Vec>::x_axis>(),
			[&](const auto k, const auto j, const auto i) {
				return bvx[k][j][i] * (dAx * i_dx) *
			           (uv[k][j][i] - uv[k][j][i - 1]);
			});

		fvmtools::apply_to_with_index(
			fvy,
			m.template full_range<topo_t<Vec>::faces, topo_t<Vec>::y_axis>(),
			[&](const auto k, const auto j, const auto i) {
				return bvy[k][j][i] * (dAy * i_dy) *
			           (uv[k][j][i] - uv[k][j - 1][i]);
			});

		fvmtools::apply_to_with_index(
			fvz,
			m.template full_range<topo_t<Vec>::faces, topo_t<Vec>::z_axis>(),
			[&](const auto k, const auto j, const auto i) {
				return bvz[k][j][i] * (dAz * i_dz) *
			           (uv[k][j][i] - uv[k - 1][j][i]);
			});
	}

	static void sum_cell_flux(topo_acc<Vec> m,
	                          field_acc_all<Vec, flecsi::ro> fu_x,
	                          field_acc_all<Vec, flecsi::ro> fu_y,
	                          field_acc_all<Vec, flecsi::ro> fu_z,
	                          field_acc_all<Vec, flecsi::wo> du) {
		const scalar_t<Vec> i_dx = 1.0 / m.template dx<topo_t<Vec>::x_axis>();
		const scalar_t<Vec> i_dy = 1.0 / m.template dx<topo_t<Vec>::y_axis>();
		const scalar_t<Vec> i_dz = 1.0 / m.template dx<topo_t<Vec>::z_axis>();

		auto fvx = m.template mdspan<topo_t<Vec>::faces>(fu_x);
		auto fvy = m.template mdspan<topo_t<Vec>::faces>(fu_y);
		auto fvz = m.template mdspan<topo_t<Vec>::faces>(fu_y);

		auto duv = m.template mdspan<topo_t<Vec>::cells>(du);

		fvmtools::apply_to_with_index(
			duv,
			m.template full_range<>(),
			[&](const auto k, const auto j, const auto i) {
				return ((fvx[k][j][i + 1] - fvx[k][j][i])) +
			           ((fvy[k][j + 1][i] - fvy[k][j][i])) +
			           ((fvz[k + 1][j][i] - fvz[k][j][i]));
			});
	}

	static void operate(topo_acc<Vec> m,
	                    scalar_t<Vec> beta,
	                    scalar_t<Vec> alpha,
	                    field_acc<Vec, flecsi::ro> a,
	                    field_acc<Vec, flecsi::ro> u,
	                    field_acc<Vec, flecsi::ro> du,
	                    field_acc<Vec, flecsi::rw> v) {
		auto av = m.template mdspan<topo_t<Vec>::cells>(a);
		auto uv = m.template mdspan<topo_t<Vec>::cells>(u);
		auto duv = m.template mdspan<topo_t<Vec>::cells>(du);
		auto vv = m.template mdspan<topo_t<Vec>::cells>(v);
		auto vol = m.volume();

		fvmtools::apply_to_with_index(
			vv,
			m.template full_range<>(),
			[&](const auto k, const auto j, const auto i) {
				return (-beta * duv[k][j][i]) +
			           (alpha * vol * av[k][j][i] * uv[k][j][i]);
			});
	}
};
} // tasks


/**
 * @brief operator of finite-volume diffusion
 *
 * linear operators of the form -β ∇ (b ∇ u ) + α a u
 * where α and β  are constants, a is a cell-centered
 * array,b is a face-centered array, and u is a cell-centered
 * array.
 *
 * @tparam Vec flecsolve::vec::msh object
 * @tparam Var (optional) specify the variable if a multivector
 */
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

template<class Vec, auto Var>
struct operator_creator<diffusion<Vec, Var>> {
	template<template<class, auto> class CoeffFn,
	         class FacesRefArr,
	         class CellsRef>
	static constexpr decltype(auto) create(FacesRefArr fra,
	                                       CellsRef cr,
	                                       scalar_t<Vec> beta,
	                                       scalar_t<Vec> alpha,
	                                       topo_slot_t<Vec> & m) {
		auto coeffop = CoeffFn<Vec, Var>::create({fra});
		auto voldiff = diffusion<Vec, Var>::create({cr, fra, beta, alpha}, m);
		return op_expr(
			flecsolve::multivariable<Vec::var.value>, coeffop, voldiff);
	}
};

}
}
