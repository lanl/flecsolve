#pragma once

#include "flecsi/execution.hh"

#include "flecsi-linalg/vectors/data/petsc.hh"
#include "petsc_tasks.hh"

namespace flecsi::linalg::vec::ops
{

struct petsc
{
	using vec_data = data::petsc;
	using real_t = PetscScalar;
	using tasks = petsc_tasks;

	template<class Other>
	void copy(const Other & x, vec_data & z) {
		execute<tasks::copy<
			typename Other::util, typename Other::topo_acc, typename Other::template acc<ro>>, mpi>(x.topo,
			                                    z.ref(), x.ref());
	}

	void zero(vec_data & x) {
		execute<tasks::set, mpi>(x.ref(), 0);
	}

	void set_to_scalar(real_t alpha, vec_data & x) {
		execute<tasks::set, mpi>(x.ref(), alpha);
	}

	void scale(real_t alpha, vec_data & x) {
		execute<tasks::scale_self, mpi>(x.ref(), alpha);
	}

	void scale(real_t alpha, const vec_data & x, vec_data & y) {
		flog_assert(x.fid() != y.fid(), "scale operation: vector data cannot be the same");
		execute<tasks::scale, mpi>(x.ref(), y.ref(), alpha);
	}
};

}
