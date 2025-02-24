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

#include "flecsolve/vectors/data/petsc.hh"

namespace flecsolve::vec::ops {

struct petsc_tasks {
	using real = PetscScalar;
	using len = std::size_t;

	template<class Util, class Topo, class OtherAcc>
	static PetscErrorCode copy(Topo m, Vec za, OtherAcc xa) {
		PetscErrorCode ierr;
		PetscScalar * arr;

		ierr = VecGetArray(za, &arr);
		CHKERRQ(ierr);
		len cnt{0};
		for (auto dof : Util::dofs(m)) {
			arr[cnt++] = xa[dof];
		}

		ierr = VecRestoreArray(za, &arr);
		CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode set(Vec x, real alpha) {
		PetscErrorCode ierr;

		ierr = VecSet(x, alpha);
		CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode scale_self(Vec x, real alpha) {
		PetscErrorCode ierr;

		ierr = VecScale(x, alpha);
		CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode scale(Vec x, Vec y, real alpha) {
		PetscErrorCode ierr;

		ierr = VecCopy(y, x);
		CHKERRQ(ierr);
		ierr = VecScale(y, alpha);
		CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode add_self(Vec z, Vec x) {
		PetscErrorCode ierr;

		ierr = VecAXPY(z, 1, x);
		CHKERRQ(ierr);

		return 0;
	}

	static PetscErrorCode add(Vec z, Vec x, Vec y) {
		PetscErrorCode ierr;

		return 0;
	}
};

}
