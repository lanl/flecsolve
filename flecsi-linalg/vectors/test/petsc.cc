#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/vectors/petsc.hh"

#include "test_mesh.hh"

namespace flecsi::linalg
{

testmesh::slot msh;
testmesh::cslot coloring;

const field<double>::definition<testmesh, testmesh::cells> mdef;
const vec::petsc::data_t::field_definition xd, yd, zd;

void init_mesh() {
	std::vector<std::size_t> extents{32};
	auto colors = testmesh::distribute(flecsi::processes(), extents);
	coloring.allocate(colors, extents);

	msh.allocate(coloring.get());
}

void init_field(testmesh::accessor<ro, ro> m,
                field<double>::accessor<wo, na> xa) {
	for (auto dof : m.dofs<testmesh::cells>()) {
		xa[dof] = m.global_id(dof);
	}
}

int vectest() {
	PetscInitialize(0, NULL, NULL, NULL);

	init_mesh();
	execute<init_field>(msh, mdef(msh));

	vec::mesh<testmesh, testmesh::cells> mvec{{mdef, msh}};
	vec::petsc x{{xd, PETSC_COMM_WORLD, mvec}}, y{{yd, PETSC_COMM_WORLD, mvec}}, z{{zd, PETSC_COMM_WORLD, mvec}};

	y.copy(mvec);

	y.set_to_scalar(3);

	y.scale(4);

	y.scale(4, x);

	PetscFinalize();

	return 0;
}

unit::driver<vectest> driver;

}
