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

#include "flecsolve/physics/boundary/bc_base.hh"
#include "flecsolve/physics/common/operator_utils.hh"
#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/common/vector_types.hh"
#include "flecsolve/physics/specializations/fvm_narray.hh"

#include "flecsolve/vectors/variable.hh"

namespace flecsolve {
namespace physics {

template<class Vec, auto Var = Vec::var.value>
struct dirichlet;

template<class Vec, auto Var>
struct operator_parameters<dirichlet<Vec, Var>> {
	scalar_t<Vec> boundary_value = 0.0;
};

template<class Vec, auto Var>
struct operator_traits<dirichlet<Vec, Var>> {
	using op_type = dirichlet<Vec, Var>;
	static constexpr std::string_view label{"dirichlet"};
};

namespace tasks {
template<class Vec, auto Axis, auto Boundary, auto Var>
struct operator_task<bc<dirichlet<Vec, Var>, Axis, Boundary>> {
	template<class U, class P>
	static constexpr void launch(const U & u, P & p) {
		const auto & subu = u.template subset(variable<Var>);
		flecsi::execute<operate>(
			subu.data.topo(), subu.data.ref(), p.boundary_value);
	}
	static constexpr void
	operate(topo_acc<Vec> m, field_acc<Vec, flecsi::wo> u, scalar_t<Vec> v) {
		fvmtools::apply_to(
			m.template mdspan<topo_t<Vec>::cells>(u),
			m.template full_range<topo_t<Vec>::cells, Axis, Boundary>(),
			[&]() { return v; });
	}
};

}
}
}
