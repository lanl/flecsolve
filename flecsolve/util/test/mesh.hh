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

	struct meta_data {
	};

	static constexpr std::size_t dimension = 1;

	template<auto>
	static constexpr std::size_t privilege_count = 2;
	template<class B>
	struct interface : B {

		template<axis A, domain DM = logical>
		std::size_t size() {
			if constexpr (DM == logical) {
				return B::template size<cells, x_axis, base::domain::logical>();
			}
			else if (DM == all) {
				return B::template size<cells, x_axis, base::domain::all>();
			}
			else if (DM == global) {
				return B::template size<cells, x_axis, base::domain::global>();
			}
		}

		FLECSI_INLINE_TARGET std::size_t global_id(std::size_t i) const {
			return i -
			       B::template offset<cells, x_axis, base::domain::logical>() +
			       B::template offset<cells, x_axis, base::domain::global>();
		}

		template<index_space Space>
		auto dofs() {
			const std::size_t start =
				B::template offset<Space, x_axis, base::domain::logical>();
			const std::size_t end = B::
				template offset<Space, x_axis, base::domain::boundary_high>();

			return flecsi::topo::make_ids<Space>(
				flecsi::util::iota_view<flecsi::util::id>(start, end));
		}
	};

	static coloring color(std::size_t num_colors, gcoord axis_extents) {
		index_definition idef;
		idef.axes =
			flecsi::topo::narray_utils::make_axes(num_colors, axis_extents);
		for (auto & a : idef.axes) {
			a.hdepth = 1;
		}

		flog_assert(idef.colors() == flecsi::processes(),
		            "current implementation is restricted to 1-to-1 mapping");

		return {{idef}};
	}

	static void initialize(flecsi::data::topology_slot<testmesh> &,
	                       coloring const &) {}
};

using realf = flecsi::field<double>;

using flecsi::na;
using flecsi::ro;
using flecsi::wo;

inline void
init_mesh(std::size_t nrows, testmesh::slot & msh, testmesh::cslot & coloring) {
	std::vector<flecsi::util::gid> extents{nrows};
	coloring.allocate(flecsi::processes(), extents);
	msh.allocate(coloring.get());
}

inline void spmv(const mat::csr<double> & A,
                 testmesh::accessor<ro> m,
                 realf::accessor<ro, na> x,
                 realf::accessor<wo, na> y) {
	auto dofs = m.dofs<testmesh::cells>();
	auto rowptr = A.offsets();
	auto colind = A.indices();
	auto values = A.values();
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
			for (std::size_t off = A.offsets()[i]; off < A.offsets()[i + 1];
			     off++) {
				if (A.indices()[off] == i) {
					out.values()[i] = 1.0 / A.values()[off];
					out.indices()[i] = i;
				}
			}
			out.offsets()[i + 1] = i + 1;
		}

		return csr_op{std::move(out)};
	}

	mat::csr<double> A;
};

}
#endif
