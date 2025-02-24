/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <utility>
#include <vector>

#include "flecsolve/physics/common/vector_types.hh"
#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/boundary/bc_base.hh"

namespace flecsolve {
namespace physics {

template<class Vec, auto Var = Vec::var.value>
struct robin;

template<class Vec, auto var>
struct operator_parameters<robin<Vec, Var>> {
	scalar_t<Vec> alpha = 1.0;
	scalar_t<Vec> beta = 0.0;
	components::faces_handle_single<Vec> faces_x;
};

namespace tasks {
template<class Vec, auto Axis, auto Boundary, auto Var>
struct operator_task<bc<robin<Vec, Var>, Axis, Boundary>> {
	template<class U, class P>
	static constexpr void launch(const U & u, P & p) {
		const auto & subu = u.template subset(variable<Var>);
		flecsi::execute<operate>(
			subu.data.topo, subu.data.ref(), p.faces_x, p.alpha, p.beta);
	}

	static void boundary_robin(topo_acc m,
	                           acc<rw> u,
	                           acc_all<ro> d,
	                           scalar_t a,
	                           scalar_t v) {
		const scalar_t<Vec> dx = m.template dx<Axis>();
		constexpr int nd = (Boundary == topo_t<Vec>::boundary_low ? -1 : 1);
		auto [jj, jo] =
			m.template get_stencil<Axis,
		                           topo_t<Vec>::cells,
		                           topo_t<Vec>::cells,
		                           Boundary>(utils::offset_seq<nd>());

		for (auto j : jj) {
			u[j] = ((2.0 * a * d[j + jo] - dx * v) /
			        (2.0 * a * d[j + jo] + dx * v)) *
			       u[j + jo];
		}
	}
};
} // tasks

}
} // namespace physics
