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

#include "flecsi/execution.hh"

#include "flecsolve/vectors/data/petsc.hh"
#include "petsc_tasks.hh"

namespace flecsolve::vec::ops {

struct petsc {
	using vec_data = data::petsc;
	using real = PetscReal;
	using scalar = PetscScalar;
	using len_t = PetscInt;
	using tasks = petsc_tasks;

	template<class Other>
	void copy(const Other & x, vec_data & z) {
		execute<tasks::copy<typename Other::util,
		                    typename Other::topo_acc,
		                    typename Other::template acc<ro>>,
		        mpi>(x.topo(), z, x.ref());
	}

	void zero(vec_data & x) { execute<tasks::set, mpi>(x, 0); }

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
		}
		else if (y == z) {
			execute<tasks::add_self, mpi>(z, y);
		}
		else {
			execute<tasks::add, mpi>(z, x, y);
		}
	}
};

}
