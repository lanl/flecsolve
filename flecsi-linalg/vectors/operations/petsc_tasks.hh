#pragma once

#include "flecsi-linalg/vectors/data/petsc.hh"

namespace flecsi::linalg::vec::ops {

struct petsc_tasks
{
	using real = PetscScalar;
	using len = std::size_t;

	template<class Util, class Topo, class OtherAcc>
	static PetscErrorCode copy(Topo m,
	                           Vec za, OtherAcc xa) {
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

	static PetscErrorCode set(Vec x, real alpha) {
		PetscErrorCode ierr;

		ierr = VecSet(x, alpha);CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode scale_self(Vec x, real alpha) {
		PetscErrorCode ierr;

		ierr = VecScale(x, alpha);CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode scale(Vec x, Vec y, real alpha) {
		PetscErrorCode ierr;

		ierr = VecCopy(y, x);CHKERRQ(ierr);
		ierr = VecScale(y, alpha);CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode add_self(Vec z, Vec x) {
		PetscErrorCode ierr;

		ierr = VecAXPY(z, 1, x);CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode add(Vec z, Vec x, Vec y) {
		PetscErrorCode ierr;

		return 0;
	}
};

}
