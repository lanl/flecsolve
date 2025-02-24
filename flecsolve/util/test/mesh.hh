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
#ifndef FLECSOLVE_UTIL_TEST_TEST_MESH_H
#define FLECSOLVE_UTIL_TEST_TEST_MESH_H

#include "flecsi/topo/narray/interface.hh"

#include "flecsolve/vectors/variable.hh"
#include "flecsolve/operators/core.hh"
#include "flecsolve/matrices/seq.hh"

namespace flecsolve {

struct testmesh : flecsi::topo::specialization<flecsi::topo::narray, testmesh> {
	enum index_space { cells };
	using index_spaces = has<cells>;
	enum domain { logical, all, global };
	enum axis { x_axis };
	using axes = has<x_axis>;
	enum boundary { low, high };
	using coord = base::coord;
	using gcoord = base::gcoord;
	using colors = base::colors;
	using hypercube = base::hypercube;
	using axis_definition = base::axis_definition;
	using index_definition = base::index_definition;

	struct meta_data {};

	static constexpr std::size_t dimension = 1;

	template<auto>
	static constexpr std::size_t privilege_count = 2;
	template<class B>
	struct interface : B {

		template<index_space Space, axis A>
		auto axis() const {
			return B::template axis<Space, A>();
		}

		FLECSI_INLINE_TARGET std::size_t global_id(std::size_t i) const {
			const auto a = axis<cells, x_axis>();
			return a.global_id(i);
		}

		template<index_space Space>
		auto dofs() const {
			const auto a = axis<Space, x_axis>();
			const std::size_t start = a.layout.template logical<0>();
			const std::size_t end = a.layout.template logical<1>();

			return flecsi::topo::make_ids<Space>(
				flecsi::util::iota_view<flecsi::util::id>(start, end));
		}
	};

	static coloring color(const index_definition & idef) { return {{idef}}; }

	static void initialize(flecsi::data::topology_slot<testmesh> &,
	                       coloring const &) {}
};

using realf = flecsi::field<double>;

using flecsi::na;
using flecsi::ro;
using flecsi::wo;

inline void init_mesh(std::size_t nrows, testmesh::slot & msh) {
	testmesh::index_definition idef;
	testmesh::gcoord extents{nrows};
	idef.axes = testmesh::base::make_axes(
		testmesh::base::distribute(flecsi::processes(), extents), extents);

	for (auto & a : idef.axes)
		a.hdepth = 1;

	msh.allocate(testmesh::mpi_coloring(idef));
}

inline void spmv(const mat::csr<double> & A,
                 testmesh::accessor<ro> m,
                 realf::accessor<ro, na> x,
                 realf::accessor<wo, na> y) {
	auto dofs = m.dofs<testmesh::cells>();
	auto [rowptr, colind, values] = A.rep();
	for (std::size_t i = 0; i < A.rows(); i++) {
		auto dof = dofs[i];
		y[dof] = 0.;
		for (std::size_t off = rowptr[i]; off < rowptr[i + 1]; off++) {
			y[dof] += x[dofs[colind[off]]] * values[off];
		}
	}
}

struct csr_op : op::base<> {

	explicit csr_op(mat::csr<double> m) : A(std::move(m)) {}

	template<class D, class R>
	void apply(const D & x, R & y) const {
		flecsi::execute<spmv, flecsi::mpi>(
			A, x.data.topo(), x.data.ref(), y.data.ref());
	}

	auto Dinv() {
		mat::csr<double> out{A.rows(), A.cols()};
		out.resize(A.nnz());

		for (std::size_t i = 0; i < A.rows(); i++) {
			for (std::size_t off = A.data.offsets()[i];
			     off < A.data.offsets()[i + 1];
			     off++) {
				if (A.data.indices()[off] == i) {
					out.data.values()[i] = 1.0 / A.data.values()[off];
					out.data.indices()[i] = i;
				}
			}
			out.data.offsets()[i + 1] = i + 1;
		}

		return csr_op{std::move(out)};
	}

	mat::csr<double> A;
};

}
#endif
