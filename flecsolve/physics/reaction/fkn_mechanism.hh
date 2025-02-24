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

#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/tasks/operator_task.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve {
namespace physics {

template<auto Var, class Scalar = double>
struct fkn_mechanism;

template<auto Var, class Scalar>
struct operator_traits<fkn_mechanism<Var, Scalar>> {
	using scalar_t = Scalar;
};

template<auto Var, class Scalar>
struct operator_parameters<fkn_mechanism<Var, Scalar>> {
	using exact_type = operator_parameters<fkn_mechanism<Var, Scalar>>;
	std::array<std::size_t, 3> indices;
	Scalar A = 0.06;
	Scalar B = 0.02;
	Scalar f = 1.0;
	Scalar kc = 1.0;
	std::array<Scalar, 4> ki = {1.28, 2.4E6, 33.6, 2.4E3};
};

namespace task {
template<class Vec, class Uec>
struct fkn_op {
	template<class Par,
	         class Topo = typename Vec::data_t::topo_t,
	         class TopoAcc = typename Vec::data_t::topo_acc,
	         class Domain = typename Vec::data_t::template acc<flecsi::ro>,
	         class Range = typename Uec::data_t::template acc<flecsi::wo>>
	static void foo(Par p, TopoAcc m, Domain f, Range g) {
		auto dofs = m.template dofs<Topo::cells>();
		auto X = dofs[p.indices[0]];
		auto Y = dofs[p.indices[1]];
		auto Z = dofs[p.indices[2]];
		g[X] = p.A * p.ki[0] * f[Y] - p.ki[1] * f[X] * f[Y] +
		       p.ki[2] * p.A * f[X] - 2 * p.ki[3] * f[X] * f[X];
		g[Y] = -p.A * p.ki[0] * f[Y] - p.ki[1] * f[X] * f[Y] +
		       0.5 * (p.kc * p.f * p.B * f[Z]);
		g[Z] = 2.0 * p.ki[2] * p.A * f[X] - p.kc * p.B * f[Z];
	}
};
}

// template<class Par, class Topo, class Field>
// void fkn(Par p, Topo m, Field f) {
// 	auto dofs = m.template dofs<Topo::cells>();
// 	f[dofs[0]] =
// 		p.A * p.ki[0] * f[dofs[1]] - p.ki[1] * f[dofs[0]] * f[dofs[1]] +
// 		p.ki[2] * p.A * f[dofs[0]] - 2 * p.ki[3] * f[dofs[0]] * f[dofs[0]];
// 	f[dofs[1]] = -p.ki[0] * p.A * f[dofs[1]] -
// 	             p.ki[1] * f[dofs[0]] * f[dofs[1]] +
// 	             0.5 * (p.kc * p.f * p.B * f[dofs[2]]);
// 	f[dofs[2]] = 2.0 * p.ki[2] * p.A * f[dofs[0]] - p.kc * p.B * f[dofs[2]];
// }

template<auto Var, class Scalar>
struct fkn_mechanism : operator_settings<fkn_mechanism<Var, Scalar>> {

	using base_type = operator_settings<fkn_mechanism<Var, Scalar>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;

	fkn_mechanism(param_type p) : base_type(p) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V & v) const {
		const auto & subu = u.template subset(variable<Var>);
		const auto & subv = v.template subset(variable<Var>);

		flecsi::execute<task::fkn_op<U, V>::template foo<param_type>>(
			this->parameters, subu.data.topo, subu.data.ref(), subv.data.ref());
	}
};

}
}