#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/solvers/krylov_interface.hh"
#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/nka.hh"

#include "csr_utils.hh"

namespace flecsolve {

testmesh::slot msh;
testmesh::cslot coloring;

const realf::definition<testmesh, testmesh::cells> xd, bd;

int nkatest() {
	UNIT () {
		auto mat = read_mm("Chem97ZtZ.mtx");
		init_mesh(mat.nrows, msh, coloring);

		csr_op A{std::move(mat)};
		vec::mesh x(msh, xd(msh)), b(msh, bd(msh));
		b.set_scalar(1.);
		x.set_scalar(3.);

		krylov_params cg_params(
			cg::settings{11, 1e-9, 1e-9, true}, cg::topo_work<>::get(b), A);
		auto P = op::create(cg_params);
		krylov_params params(nka::settings{100, 0., 1e-6, 5, 0.2},
		                     nka::topo_work<5>::get(b),
		                     A,
		                     P);
		auto slv = op::create(params);
		auto info = slv.apply(b, x);
		EXPECT_EQ(info.iters, 17);
	};

	return 0;
}

flecsi::unit::driver<nkatest> driver;
}
