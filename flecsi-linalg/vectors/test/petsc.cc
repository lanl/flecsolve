#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/vectors/petsc.hh"

namespace flecsi::linalg
{

void create_petscvec(Vec * v) {
	VecCreateMPI(MPI_COMM_WORLD, PETSC_DECIDE, 32, v);
	VecSet(*v, 4);
}

int vectest() {
	PetscInitialize(0, NULL, NULL, NULL);

	Vec v;
	execute<create_petscvec, mpi>(&v);

	vec::petsc x{{v}};

	PetscFinalize();

	return 0;
}

unit::driver<vectest> driver;

}
