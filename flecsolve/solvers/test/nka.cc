#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/util/config.hh"
#include "flecsolve/solvers/krylov_operator.hh"
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

		cg::settings pre_settings("preconditioner");
		nka::settings nnl_settings("solver");
		read_config("nka.cfg", pre_settings, nnl_settings);

		op::krylov P(op::krylov_parameters(
			pre_settings, cg::topo_work<>::get(b), std::ref(A)));

		op::krylov slv(op::krylov_parameters(nnl_settings,
		                                     nka::topo_work<5>::get(b),
		                                     std::ref(A),
		                                     std::move(P)));
		auto info = slv.apply(b, x);
		EXPECT_EQ(info.iters, 17);
	};

	return 0;
}

flecsi::unit::driver<nkatest> driver;
}
