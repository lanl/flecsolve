#pragma once

#include "flecsi/execution.hh"

#include "flecsi-linalg/vectors/data/petsc.hh"
#include "petsc_tasks.hh"

namespace flecsi::linalg::vec::ops
{

struct petsc
{
	using vec_data = data::petsc;
	using real = PetscReal;
	using scalar = PetscScalar;
	using len_t = PetscInt;
	using tasks = petsc_tasks;

	template<class Other>
	void copy(const Other & x, vec_data & z) {
		execute<tasks::copy<
			typename Other::util, typename Other::topo_acc, typename Other::template acc<ro>>, mpi>(x.topo,
				                  z, x.ref());
	}

	void zero(vec_data & x) {
		execute<tasks::set, mpi>(x, 0);
	}

	void set_to_scalar(scalar alpha, vec_data & x) {
		execute<tasks::set, mpi>(x, alpha);
	}

	void scale(scalar alpha, vec_data & x) {
		execute<tasks::scale_self, mpi>(x, alpha);
	}

	void scale(scalar alpha, const vec_data & x, vec_data & y) {
		flog_assert(x != y, "scale operation: vector data cannot be the same");
		execute<tasks::scale, mpi>(x, y, alpha);
	}

	void add(const vec_data & x, const vec_data & y, vec_data & z) {
		if (x == z) {
			execute<tasks::add_self, mpi>(z, y);
		} else if (y == z) {
			execute<tasks::add_self, mpi>(z, y);
		} else {
			execute<tasks::add, mpi>(z, x, y);
		}
	}
};

}
