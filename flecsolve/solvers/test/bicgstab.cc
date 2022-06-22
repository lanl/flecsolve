#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/solvers/bicgstab.hh"
#include "flecsolve/util/config.hh"

#include "csr_utils.hh"

namespace flecsolve {

static constexpr std::size_t ncases = 2;
std::array<testmesh::slot, ncases> mshs;
std::array<testmesh::cslot, ncases> colorings;

const realf::definition<testmesh, testmesh::cells> xd, bd;

int driver() {
	std::array cases{std::make_tuple("Chem97ZtZ.mtx", 91, 92),
	                 std::make_tuple("psmigr_3.mtx", 32, 32)};

	static_assert(cases.size() <= ncases);

	UNIT () {
		bicgstab::settings settings("solver");
		read_config("bicgstab.cfg", settings);

		std::size_t i = 0;
		for (const auto & cs : cases) {
			auto mat = read_mm(std::get<0>(cs));

			auto & msh = mshs[i];

			init_mesh(mat.nrows, msh, colorings[i]);

			csr_op A{std::move(mat)};

			vec::mesh x(msh, xd(msh)), b(msh, bd(msh));
			b.set_random(0);
			x.set_random(1);

			op::krylov slv(op::krylov_parameters(
				settings, bicgstab::topo_work<>::get(b), std::move(A)));

			auto info = slv.apply(b, x);

			EXPECT_EQ(info.status, solve_info::stop_reason::converged_rtol);
			EXPECT_TRUE((info.iters == std::get<1>(cs)) ||
			            (info.iters == std::get<2>(cs)));

			++i;
		}
	};
}

flecsi::unit::driver<driver> drv;

}
