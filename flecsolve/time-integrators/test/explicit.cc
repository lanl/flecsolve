#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/time-integrators/rk23.hh"
#include "flecsolve/util/config.hh"

#include "test_mesh.hh"

namespace flecsolve {

testmesh::slot msh;
testmesh::cslot coloring;

const flecsi::field<double>::definition<testmesh, testmesh::cells> xd, bd;

static void
init_mesh(std::size_t nrows, testmesh::slot & msh, testmesh::cslot & coloring) {
	std::vector<std::size_t> extents{nrows};
	auto colors = testmesh::distribute(flecsi::processes(), extents);
	coloring.allocate(colors, extents);
	msh.allocate(coloring.get());
}

int extest() {
	using namespace flecsolve::time_integrator;

	init_mesh(32, msh, coloring);

	vec::mesh x(msh, xd(msh)), b(msh, bd(msh));

	rk23::parameters params("time-int", op::I, rk23::topo_work<>());
	read_config("config.cfg", params);

	rk23::integrator ti(params);

	return 0;
}

flecsi::unit::driver<extest> driver;
}
