#pragma once

#include "flecsi-linalg/vectors/data/petsc.hh"

namespace flecsi::linalg::vec::ops {

struct petsc_tasks
{
	using real = data::petsc::real_t;
	using len = data::petsc::len_t;
	template<flecsi::partition_privilege_t priv>
	using acc = data::petsc::acc<priv>;

	template<class Util, class Topo, class OtherAcc>
	static PetscErrorCode copy(Topo m,
	                           acc<wo> za, OtherAcc xa) {
		PetscErrorCode ierr;
		PetscScalar *arr;

		ierr = VecGetArray(za, &arr);CHKERRQ(ierr);
		len cnt{0};
		for (auto dof : Util::dofs(m)) {
			arr[cnt++] = xa[dof];
		}

		ierr = VecRestoreArray(za, &arr);CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode set(acc<wo> xa, real alpha) {
		PetscErrorCode ierr;

		ierr = VecSet(xa, alpha);CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode scale_self(acc<wo> xa, real alpha) {
		PetscErrorCode ierr;

		ierr = VecScale(xa, alpha);CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode scale(acc<ro> x, acc<wo> y, real alpha) {
		PetscErrorCode ierr;

		ierr = VecCopy(y, x);CHKERRQ(ierr);
		ierr = VecScale(y, alpha);CHKERRQ(ierr);

		return 0;
	}
};

}
