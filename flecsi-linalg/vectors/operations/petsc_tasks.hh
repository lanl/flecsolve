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
};

}
