#pragma once

#include "flecsi/execution.hh"

#include "flecsi-linalg/vectors/data/petsc_data.hh"
#include "petsc_tasks.hh"

namespace flecsi::linalg
{

struct petsc_operations
{
	using vec_data = petsc_data;
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
