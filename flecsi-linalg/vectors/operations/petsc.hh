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
			typename Other::util, typename Other::topo_acc, typename Other::ro_acc>, mpi>(x.topo,
			                                    z.ref(), x.ref());
	}
};

}
