#pragma once

#include <cstddef>
#include <flecsi/util/mpi.hh>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "flecsolve/physics/common/operator_utils.hh"
#include "flecsi/data.hh"

namespace flecsolve {
namespace physics {
namespace tasks {

using flecsi::na;
using flecsi::ro;
using flecsi::rw;
using flecsi::wo;

template<class Topo, class Field>
struct topology_tasks {
	using topo_t = Topo;
	using topo_acc = typename topo_t::template accessor<ro>;
	using index_space = typename topo_t::index_space;
	using axes = typename topo_t::axes;
	using axis = typename topo_t::axis;
	using domain = typename topo_t::domain;
	using scalar_t = typename Field::value_type;
	static constexpr std::size_t dim = axes::size;

	static inline constexpr flecsi::PrivilegeCount num_priv =
		topo_t::template privilege_count<0>;

	template<flecsi::partition_privilege_t priv>
	static inline constexpr flecsi::Privileges dofs_priv =
		flecsi::privilege_cat<
			flecsi::privilege_repeat<priv, num_priv - (num_priv > 1)>,
			flecsi::privilege_repeat<na, (num_priv > 1)>>;

	template<flecsi::partition_privilege_t priv>
	using acc = typename Field::template accessor1<dofs_priv<priv>>;

	template<flecsi::partition_privilege_t priv>
	using acc_all = typename Field::template accessor1<
		flecsi::privilege_repeat<priv, num_priv>>;

	template<std::size_t I>
	static constexpr auto cast_idx() {
		return static_cast<axis>(I);
	}

	template<auto Axis, std::size_t I>
	static constexpr auto jump_idx() {
		return static_cast<axis>((Axis + I) % dim);
	}

	template<auto Axis>
	static constexpr auto next_idx() {
		return jump_idx<Axis, 1>();
	}

	/**
	 * @brief Rotates memory of input field (aligned w/ `From`) into `To.
	 *
	 * @tparam From axis to scan through `u`
	 * @tparam To axis to send to
	 * @param m topology accessor
	 * @param u cell-centered field with input
	 * @param uy cell-centered file to send to
	 */
	// template<axis From, axis To>
	// static void rotate_to(topo_acc m, acc_all<ro> u, acc_all<rw> uy)
	// {
	//   auto jj = m.template get_idxs<To>();
	//   auto kk = m.template get_idxs<From, To>();

	//   for (int i = 0; i < jj.size(); ++i) {
	//     uy[jj[i]] = u[kk[i]];
	//   }
	// }

	/**
	 * @brief Apply a stencil kernel along axis A
	 *
	 * @tparam K kernel functor; the interface is under development
	 * @tparam A axis
	 * @tparam From the index space to pull from, e.g. the domain
	 * @tparam To the index space of the result, e.g. the range
	 * @param[in] m topology accessor
	 * @param[in] u field with the index space From
	 * @param[out] fu field with index space To
	 */
	template<class K, axis A, index_space From, index_space To = From>
	static void operate_kernel(topo_acc m, acc_all<ro> u, acc_all<rw> fu) {
		auto [jj, jm1, jp1] =
			m.template get_stencil<A, From, To>(utils::offset_seq<-1, 1>());

		for (auto j : jj) {
			fu[j] = K::op(u[j], u[j + jm1], 1.0);
		}
	}

	/**
	 * @brief blanks a field
	 *
	 * @param m topology accessor
	 * @param u field to zero
	 */
	static void zero_op(topo_acc m, acc_all<wo> u) {
		auto jj = m.template get_stencil<topo_t::x_axis, topo_t::cells>(
			utils::offset_seq<>());
		for (auto j : jj) {
			u[j] = 0.0;
		}
	}

	/**
	 * @brief calculates the face-centered flux component along axis A, using
	 * the gradient and a diffusion coefficent
	 *
	 * @tparam A axis of flux component to calculuate
	 * @param m topology accessor
	 * @param u_x cell-centered value
	 * @param b_x face-centered directional difffusion coefficent
	 * @param fu_x face-centered field for flux component along axis A
	 */
	template<axis A>
	static void
	flux_op(topo_acc m, acc_all<ro> u_x, acc<ro> b_x, acc<rw> fu_x) {
		const scalar_t dA = m.template normal_dA<A>();
		const scalar_t idx = 1.0 / m.template dx<A>();

		auto [jj, jm1] =
			m.template get_stencil<A, topo_t::cells, topo_t::faces>(
				utils::offset_seq<-1>());

		for (auto j : jj) {
			fu_x[j] = b_x[j] * (dA * idx) * (u_x[j] - u_x[j + jm1]);
		}
	}

	template<axis A>
	static void flux_op_1(topo_acc m, acc<ro> u_x, acc<rw> fu_x) {
		const scalar_t dA = m.template normal_dA<A>();
		const scalar_t idx = 1.0 / m.template dx<A>();

		auto [jj, jm1] =
			m.template get_stencil<A, topo_t::cells, topo_t::faces>(
				utils::offset_seq<-1>());

		for (auto j : jj) {
			fu_x[j] = (dA * idx) * (u_x[j] - u_x[j + jm1]);
		}
	}
	/**
	 * @brief sums the face-centered flux components into cell-centered quantity
	 * change. This function is expected to be called on all axis to produce a
	 * total quantity change of the fluxes. Note that du should be zero'd before
	 * iterating over the axis.
	 *
	 * @tparam A axis to accumlate
	 * @param m topology accessor
	 * @param fu_x face-centered field of flux components along axis A
	 * @param du cell-centered field with quantity change.
	 */
	template<axis A>
	static void flux_sum(topo_acc m, acc_all<ro> fu_x, acc<rw> du) {
		const scalar_t i_dx = 1.0 / m.template dx<A>();
		auto [jj, jp1] =
			m.template get_stencil<A, topo_t::cells>(utils::offset_seq<1>());

		for (auto j : jj) {
			du[j] += (fu_x[j + jp1] - fu_x[j]) * i_dx;
		}
	}

	/**
	 * @brief perform a full diffusion apply, with both the surface flux and any
	 * additional source terms.
	 *
	 * @param m topology accessor
	 * @param beta scalar coefficent of diffusive term
	 * @param alpha scalar coefficent of source terms
	 * @param a cell-centered field of source term coefficents
	 * @param u cell-centered field being operated on
	 * @param du cell-centered field of integrated surface fluxes
	 * @param un cell-centered field after operation
	 */
	static void diffuse_op(topo_acc m,
	                       scalar_t beta,
	                       scalar_t alpha,
	                       acc_all<ro> a,
	                       acc_all<ro> u,
	                       acc_all<ro> du,
	                       acc<rw> un) {
		auto jj = m.template get_stencil<topo_t::x_axis, topo_t::cells>(
			utils::offset_seq<>());

		for (auto j : jj) {
			// un[j] = std::max(0.0, (-beta * du[j]) + (alpha * a[j] * u[j]));
			un[j] = (-beta * du[j]) + (alpha * a[j] * u[j]);
		}
	}

	static void diffuse_op_1(topo_acc m,
	                         scalar_t beta,
	                         scalar_t alpha,
	                         acc_all<ro> u,
	                         acc_all<ro> du,
	                         acc<rw> un) {
		auto jj = m.template get_stencil<topo_t::x_axis, topo_t::cells>(
			utils::offset_seq<>());

		for (auto j : jj) {
			// un[j] = std::max(0.0, (-beta * du[j]) + (alpha * a[j] * u[j]));
			un[j] = (-beta * du[j]);
		}
	}
	/**
	 * @brief simple boundary operation, used by dirchilet operator
	 *
	 * @tparam A boundary axis
	 * @tparam D boundary domain (e.g. low, high)
	 * @param v scalar value to set boundary zones to
	 * @param m topology accessor
	 * @param u cell-centered field which operation is applied on
	 */
	template<axis A, domain D>
	static void boundary_dirichlet(topo_acc m, acc<wo> u, scalar_t v) {
		auto jj = m.template get_stencil<A, topo_t::cells, topo_t::cells, D>(
			utils::offset_seq<>());

		for (auto j : jj) {
			u[j] = v;
		}
	}

	/**
	 * @brief zero-flux boundary operation, used by neumann operator
	 *
	 * @tparam A boundary axis
	 * @tparam D boundary domain (e.g. low, high)
	 * @param m topology accessor
	 * @param u cell-centered field which operation is applied on
	 */
	template<axis A, domain D>
	static void boundary_neumann(topo_acc m, acc<rw> u, scalar_t v) {
		const scalar_t dx = m.template dx<A>();
		constexpr int nd = (D == topo_t::boundary_low ? 1 : -1);
		auto [jj, jo] =
			m.template get_stencil<A, topo_t::cells, topo_t::cells, D>(
				utils::offset_seq<nd>());

		for (auto j : jj) {
			u[j] = u[j + jo] - static_cast<scalar_t>(nd) * dx * v;
		}
	}

	/**
	 * @brief mixed boundary operation, used by robin operator
	 *
	 * @tparam A boundary axis
	 * @tparam D boundary domain (e.g. low, high)
	 * @param m topology accessor
	 * @param u cell-centered field which operation is applied on
	 */
	template<axis A, domain D>
	static void boundary_robin(topo_acc m,
	                           acc<rw> u,
	                           acc_all<ro> d,
	                           scalar_t a,
	                           scalar_t v) {
		const scalar_t dx = m.template dx<A>();
		constexpr int nd = (D == topo_t::boundary_low ? -1 : 1);
		auto [jj, jo] =
			m.template get_stencil<A, topo_t::cells, topo_t::cells, D>(
				utils::offset_seq<nd>());

		for (auto j : jj) {
			u[j] = ((2.0 * a * d[j + jo] - dx * v) /
			        (2.0 * a * d[j + jo] + dx * v)) *
			       u[j + jo];
		}
	}

	template<axis A, domain D>
	static void
	boundary_robin_1(topo_acc m, acc<rw> u, scalar_t a, scalar_t v) {
		const scalar_t dx = m.template dx<A>();
		constexpr int nd = (D == topo_t::boundary_low ? -1 : 1);
		auto [jj, jo] =
			m.template get_stencil<A, topo_t::cells, topo_t::cells, D>(
				utils::offset_seq<nd>());

		for (auto j : jj) {
			u[j] = ((2.0 * a - dx * v) / (2.0 * a + dx * v)) * u[j + jo];
		}
	}

	template<axis A>
	static void flux_limited_diffusion(topo_acc m,
	                                   acc_all<ro> u_x,
	                                   acc<wo> b_x,
	                                   scalar_t Z,
	                                   scalar_t alpha,
	                                   scalar_t beta) {
		// if constexpr (A == topo_t::z_axis) {
		// 	return;
		// }
		const scalar_t i_dx = 1.0 / m.template dx<A>();
		auto [jj, jm1] =
			m.template get_stencil<A, topo_t::cells, topo_t::faces>(
				utils::offset_seq<-1>());

		for (auto j : jj) {
			auto uder = i_dx * (u_x[j] - u_x[j + jm1]);
			auto uavg = 0.5 * (u_x[j] + u_x[j + jm1]);
			b_x[j] = 1.; // / (1. / std::abs(uder) / uavg);
		}
	}
};

}
}
} // namespace tasks
