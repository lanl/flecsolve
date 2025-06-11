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
#ifndef FLECSOLVE_EXAMPLES_POISSON_POISSON_HH
#define FLECSOLVE_EXAMPLES_POISSON_POISSON_HH

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"

#include "flecsolve/operators/core.hh"

#include "mesh.hh"

namespace poisson {

struct parameters {
	using ref_t = poisson::template stencil_field<
		poisson::five_pt>::Reference<poisson::mesh, poisson::mesh::vertices>;
	ref_t op_reference;
};

struct poisson_op : flecsolve::op::base<parameters> {
	using base = flecsolve::op::base<parameters>;
	using base::params;

	poisson_op(parameters::ref_t ref) : base(ref) {}

	template<class D, class R>
	void apply(const D & x, R & y) const {
		flecsi::execute<spmv>(
			y.data.topo(), params.op_reference, y.data.ref(), x.data.ref());
	}

	auto ref() { return params.op_reference; }

protected:
	static void spmv(mesh::accessor<ro> m,
	                 stencil_field<five_pt>::accessor<ro, ro> soa,
	                 field<double>::accessor<wo, na> ya,
	                 field<double>::accessor<ro, ro> xa) {

		auto y = m.mdcolex<mesh::vertices>(ya);
		auto x = m.mdcolex<mesh::vertices>(xa);
		const auto so = m.stencil_op<mesh::vertices, five_pt>(soa);

		for (auto j : m.vertices<mesh::y_axis>()) {
			for (auto i : m.vertices<mesh::x_axis>()) {
				y(i, j) = so(i, j, five_pt::c) * x(i, j) -
				          so(i, j, five_pt::w) * x(i - 1, j) -
				          so(i + 1, j, five_pt::w) * x(i + 1, j) -
				          so(i, j, five_pt::s) * x(i, j - 1) -
				          so(i, j + 1, five_pt::s) * x(i, j + 1);
			}
		}
	}
};

}

#endif
