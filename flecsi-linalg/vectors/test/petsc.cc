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

const field<double>::definition<testmesh, testmesh::cells> xd;
const vec::petsc::data_t::field_definition yd;

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
	using mesh_vec = vec::mesh<testmesh, testmesh::cells>;
	mesh_vec x{{xd, msh}};
	vec::petsc y{{yd, MPI_COMM_WORLD, x}};

	y.copy(x);

	return 0;
}

unit::driver<vectest> driver;

}
