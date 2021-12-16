#pragma once

#include "flecsi-linalg/vectors/data/petsc_data.hh"

namespace flecsi::linalg {

struct petsc_tasks
{
	using real = petsc_data::real_t;
	using len = petsc_data::len_t;
	template<flecsi::partition_privilege_t priv>
	using acc = petsc_data::acc<priv>;

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
