#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"


#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/operators/bicgstab.hh"


#include "csr_utils.hh"

namespace flecsi::linalg {

testmesh::slot msh;
testmesh::cslot coloring;

const realf::definition<testmesh, testmesh::cells> xd, bd;


int driver() {
	auto mat = read_mm("Chem97ZtZ.mtx");

	init_mesh(mat.nrows, msh, coloring);

	csr_op A{std::move(mat)};

	vec::mesh x(msh, xd(msh)), b(msh, bd(msh));
	b.set_random();
	x.set_random();

	bicgstab::solver slv(bicgstab::default_settings(),
	                     bicgstab::topo_work<>::get(b));

	slv.settings.maxiter = 200;
	std::size_t i = 0;
	slv.apply(A, b, x, [&](const auto &, double rnorm) {
		std::cout << i++ << " " << rnorm << std::endl;
	});

	return 0;
}


unit::driver<driver> drv;

}
