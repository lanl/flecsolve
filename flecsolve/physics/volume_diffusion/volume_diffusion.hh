#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <flecsi/util/array_ref.hh>
#include <flecsi/util/constant.hh>
#include <tuple>
#include <utility>

#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/common/operator_utils.hh"
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
template<auto Var, class Vec>
struct volume_diffusion_op;

// template<auto Var, class Topo, class Scalar>
// struct operator_traits<volume_diffusion_op<Var, Topo, Scalar>> {
// 	using scalar_t = Scalar;
// 	using topo_t = Topo;
// 	using topo_slot_t = flecsi::data::topology_slot<Topo>;
// 	using topo_axes_t = typename topo_t::axes;
// 	constexpr static auto dim = Topo::dimension;
// 	using tasks_f = tasks::topology_tasks<topo_t, flecsi::field<scalar_t>>;

// 	using cell_def =
// 		typename flecsi::field<scalar_t>::template definition<topo_t,
// 	                                                          topo_t::cells>;
// 	using cell_ref =
// 		typename flecsi::field<scalar_t>::template Reference<topo_t,
// 	                                                         topo_t::cells>;

// 	using face_def =
// 		typename flecsi::field<scalar_t>::template definition<topo_t,
// 	                                                          topo_t::faces>;
// 	using face_ref =
// 		typename flecsi::field<scalar_t>::template Reference<topo_t,
// 	                                                         topo_t::faces>;
// };

template<auto Var, class Vec>
struct operator_parameters<volume_diffusion_op<Var, Vec>>
	: components::CellsHandle<Vec>, components::FacesHandle<Vec> {
	using op_type = volume_diffusion_op<Var, Vec>;

	scalar_t<Vec> beta = 1.0;
	scalar_t<Vec> alpha = 0.0;
};

namespace tasks {
template<auto Var, class Vec>
struct operator_task<volume_diffusion_op<Var, Vec>> {
	template<auto Axis>
	static void update_flux(topo_acc<Vec> m,
	                                field_acc_all<Vec, flecsi::ro> u_x,
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
	                                  field_acc_all<Vec, flecsi::ro> fu_x,
	                                  field_acc<Vec, rw> du) {
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
	                    field_acc<Vec, rw> un) {
		auto jj =
			m.template get_stencil<topo_t<Vec>::x_axis, topo_t<Vec>::cells>(
				utils::offset_seq<>());

		for (auto j : jj) {
			un[j] = (-beta * du[j]) + (alpha * a[j] * u[j]);
		}
	}

	static void zero_op(topo_acc<Vec> m, field_acc_all<Vec, wo> u) {
		auto jj = m.template get_stencil<topo_t<Vec>::x_axis, topo_t<Vec>::cells>(
			utils::offset_seq<>());
		for (auto j : jj) {
			u[j] = 0.0;
		}
	}
};
}

template<auto Var, class Vec>
struct volume_diffusion_op : operator_settings<volume_diffusion_op<Var, Vec>> {

	using base_type = operator_settings<volume_diffusion_op<Var, Vec>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;
	using task_type = typename base_type::task_type;

	// using scalar_t = typename operator_traits<exact_type>::scalar_t;
	// using topo_t = typename operator_traits<exact_type>::topo_t;

	// // using diffop_t = DiffOp<my_t>;
	// using topo_slot_t = typename operator_traits<exact_type>::topo_slot_t;
	// using topo_axes_t = typename operator_traits<exact_type>::topo_axes_t;
	// using cell_def = typename operator_traits<exact_type>::cell_def;
	// using cell_ref = typename operator_traits<exact_type>::cell_ref;
	// using face_def = typename operator_traits<exact_type>::face_def;
	// using face_ref = typename operator_traits<exact_type>::face_ref;


	static constexpr std::size_t dim = topo_axes_t<Vec>::size;

	using flux_store_t =
		common::topo_state_store<exact_type, Vec, topo_t<Vec>::faces, dim, 0>;
	// zero-D quantity change due to flux integration on surface
	using du_store_t =
		common::topo_state_store<exact_type, Vec, topo_t<Vec>::cells, 1, 0>;

	// These arrays hold references to the fields provided by the above
	// declarations
	std::array<face_ref<Vec>, dim> fluxes;

	cell_ref<Vec> du;

	volume_diffusion_op(topo_slot_t<Vec> & s, param_type parameters)
		: base_type(parameters), fluxes(flux_store_t::get_state(s)),
		  du(du_store_t::get_state(s)) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V & v) const {
		auto & subu = u.template subset(variable<Var>);
		auto & subv = v.template subset(variable<Var>);

		_apply(subu.data.topo, subu.data.ref(), subv.data.ref());
	}

	void _apply(topo_slot_t<Vec> & m, cell_ref<Vec> u, cell_ref<Vec> v) const {
		// first, zero-out the fields to take results
		flecsi::execute<task_type::zero_op>(m, v);
		flecsi::execute<task_type::zero_op>(m, du);

		// determine the fluxes along the axis
		sweep(m, u, topo_axes_t<Vec>());

		// collect all prior calculations and apply to range vector

		flecsi::execute<task_type::operate>(m,
		                                     this->parameters.beta,
		                                     this->parameters.alpha,
		                                     *(this->parameters.cells),
		                                     u,
		                                     du,
		                                     v);
	}

	template<auto... Axis>
	constexpr void
	sweep(topo_slot_t<Vec> & m, cell_ref<Vec> u, flecsi::util::constants<Axis...>) const {

		(flecsi::execute<task_type::template update_flux<Axis>>(
			 m, u, (*(this->parameters.faces))[Axis], fluxes[Axis]),
		 ...);

		(flecsi::execute<task_type::template sum_cell_flux<Axis>>(m, fluxes[Axis], du),
		 ...);
	}
};

}
}
