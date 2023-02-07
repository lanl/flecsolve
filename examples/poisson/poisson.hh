#ifndef FLECSOLVE_EXAMPLES_POISSON_POISSON_HH
#define FLECSOLVE_EXAMPLES_POISSON_POISSON_HH

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"

#include "flecsolve/operators/base.hh"

#include "mesh.hh"

namespace poisson {
struct poisson_op;
}

template<>
struct flecsolve::op::traits<poisson::poisson_op> {
	struct parameters {
		using ref_t = poisson::template stencil_field<poisson::five_pt>::Reference<poisson::mesh, poisson::mesh::vertices>;
		ref_t op_reference;
	};

	static constexpr auto input_var = flecsolve::variable<flecsolve::anon_var::anonymous>;
	static constexpr auto output_var = flecsolve::variable<flecsolve::anon_var::anonymous>;
};



namespace poisson {

struct poisson_op : flecsolve::op::base<poisson_op> {
	using base = flecsolve::op::base<poisson_op>;
	using base::params;

	template<class D, class R>
	void apply(const D & x, R & y) const {
		flecsi::execute<spmv>(
			y.data.topo(), params.op_reference, y.data.ref(), x.data.ref());
	}

	auto ref() {
		return params.op_reference;
	}

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
				y(i,j) = so(i, j, five_pt::c) * x(i, j)
					- so(i,   j  , five_pt::w) * x(i-1, j  )
					- so(i+1, j  , five_pt::w) * x(i+1, j  )
					- so(i,   j  , five_pt::s) * x(i,   j-1)
					- so(i,   j+1, five_pt::s) * x(i,   j+1);
			}
		}
	}
};

}

#endif
