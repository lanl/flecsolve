#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"


#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/operators/bicgstab.hh"


#include "csr_utils.hh"

namespace flecsi::linalg {

static constexpr std::size_t ncases = 2;
std::array<testmesh::slot, ncases> mshs;
std::array<testmesh::cslot, ncases> colorings;

const realf::definition<testmesh, testmesh::cells> xd, bd;

int driver() {
	std::array cases{
		std::make_pair("Chem97ZtZ.mtx", 91),
		std::make_pair("psmigr_3.mtx", 32)
	};

	static_assert(cases.size() <= ncases);

	UNIT() {
		std::size_t i = 0;
		for (const auto & cs : cases) {
			auto mat = read_mm(cs.first);

			auto & msh = mshs[i];

			init_mesh(mat.nrows, msh, colorings[i]);

			csr_op A{std::move(mat)};

			vec::mesh x(msh, xd(msh)), b(msh, bd(msh));
			b.set_random(0);
			x.set_random(1);

			bicgstab::solver slv(bicgstab::settings{200, 1e-9, false},
			                     bicgstab::topo_work<>::get(b));

			auto info = slv.apply(A, b, x);

			EXPECT_EQ(info.status, solve_info::stop_reason::converged_rtol);
			EXPECT_EQ(info.iters, cs.second);

			++i;
		}
	};
}


unit::driver<driver> drv;

}
