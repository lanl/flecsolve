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
#include <flecsi/util/array_ref.hh>
#include <iterator>
#include <utility>
#include <vector>
#include <list>

#include "flecsolve/physics/common/operator_base.hh"
#include "flecsolve/physics/tasks/operator_task.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve {
namespace physics {

template<auto Var, std::size_t N, class Scalar = double>
struct arrhenius;

template<auto Var, std::size_t N, class Scalar>
struct operator_traits<arrhenius<Var, N, Scalar>> {
	using scalar_t = Scalar;
};

template<auto Var, std::size_t N, class Scalar>
struct operator_parameters<arrhenius<Var, N, Scalar>> {
	using exact_type = operator_parameters<arrhenius<Var, N, Scalar>>;
	static constexpr auto nr = N;
	Scalar T;
	// std::vector<std::size_t> s_idx;
	std::array<std::vector<std::size_t>, N> rea_idx;
	std::array<std::vector<std::size_t>, N> pro_idx;
	std::array<Scalar, N> alpha = utils::make_array<Scalar, N>(1.0);
	std::array<Scalar, N> beta = utils::make_array<Scalar, N>(0.0);
	std::array<Scalar, N> gamma = utils::make_array<Scalar, N>(0.0);
};

template<class Vec, class Uec>
struct arr_op {
	template<class Par,
	         class Topo = typename Vec::data_t::topo_t,
	         class TopoAcc = typename Vec::data_t::topo_acc,
	         class Domain = typename Vec::data_t::template acc<flecsi::ro>,
	         class Range = typename Uec::data_t::template acc<flecsi::wo>>
	static void apply(Par p, TopoAcc m, Domain f, Range g) {
		auto dofs = m.template dofs<Topo::cells>();
		for (std::size_t a = 0; a < Par::nr; ++a) {
			const auto rate = p.alpha[a] * std::pow(p.T / 300., p.beta[a]) *
			                  std::exp(-p.gamma[a] / p.T);

			double fi = 1.0;
			for (auto i : p.rea_idx[a]) {
				fi *= f[dofs[i]];
			}
			fi *= rate;
			for (auto i : p.rea_idx[a]) {
				g[dofs[i]] -= fi;
			}
			for (auto i : p.pro_idx[a]) {
				g[dofs[i]] += fi;
			}
		}
	}

	template<class Par,
	         class Scalar = typename Vec::real,
	         class Topo = typename Vec::data_t::topo_t,
	         class TopoAcc = typename Vec::data_t::topo_acc,
	         class Domain = typename Vec::data_t::template acc<flecsi::ro>,
	         class Range = typename Uec::data_t::template acc<flecsi::wo>>
	static void jacobian(Par p, TopoAcc m, Domain f, Range g) {

		const std::size_t ns = p.s_idx.size();
		auto dofs = m.template dofs<Topo::cells>();
		auto J = utils::make_array<Scalar, ns * ns>(0.0);
		auto Jview = flecsi::util::mdspan(J.data());

		for (std::size_t a = 0; a < Par::nr; ++a) {
			const auto rate = p.alpha[a] * std::pow(p.T / 300., p.beta[a]) *
			                  std::exp(-p.gamma[a] / p.T);
			double fi;
			for (auto r : p.rea_idx[a]) {
				fi = rate;

				for (auto r2 : p.rea_idx[a]) {
					if (r != r2)
						fi *= f[dofs[r2]];
				}

				for (auto r2 : p.rea_idx[a]) {
					J[r2][r] -= fi;
				}

				for (auto pr : p.pro_idx[a]) {
					J[p][r] += fi;
				}
			}
		}
		// for (std::size_t s = 0; s < ns; ++s) {
		// 	for (std::size_t s2 = 0; s2 < ns; ++s2) {
		// 		g[dofs[p.s_idx[0]]] += 0.0; //????
		// 	}
		// }
	}
};

template<auto Var, std::size_t N, class Scalar>
struct arrhenius : operator_settings<arrhenius<Var, N, Scalar>> {

	using base_type = operator_settings<arrhenius<Var, N, Scalar>>;
	using exact_type = typename base_type::exact_type;
	using param_type = typename base_type::param_type;

	arrhenius(param_type p) : base_type(p) {}

	template<class U, class V>
	constexpr auto apply(const U & u, V & v) const {
		const auto & subu = u.template subset(variable<Var>);
		const auto & subv = v.template subset(variable<Var>);

		flecsi::execute<arr_op<U, V>::template apply<param_type>>(
			this->parameters, subu.data.topo, subu.data.ref(), subv.data.ref());
	}
};
}
}