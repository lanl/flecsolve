#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/solvers/bicgstab.hh"
#include "flecsolve/util/config.hh"
#include "flecsolve/matrices/io/matrix_market.hh"

#include "flecsolve/util/test/mesh.hh"

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
		auto settings =
			read_config("bicgstab.cfg", bicgstab::options("solver"));
		std::size_t i = 0;
		for (const auto & cs : cases) {
			auto mtx = mat::io::matrix_market<>::read(std::get<0>(cs)).tocsr();

			auto & msh = mshs[i];

			init_mesh(mtx.rows(), msh, colorings[i]);

			auto [x, b] = vec::make(msh)(xd, bd);
			b.set_random(0);
			x.set_random(1);

			op::krylov slv(
				op::krylov_parameters(settings,
			                          bicgstab::topo_work<>::get(b),
			                          op::core<csr_op>(std::move(mtx))));

			auto info = slv.apply(b, x);

			EXPECT_EQ(info.status, solve_info::stop_reason::converged_rtol);
			EXPECT_TRUE((info.iters == std::get<1>(cs)) ||
			            (info.iters == std::get<2>(cs)));

			++i;
		}
	};
}

flecsi::util::unit::driver<driver> drv;

}
