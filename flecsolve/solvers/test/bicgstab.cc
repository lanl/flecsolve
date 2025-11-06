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

const realf::definition<testmesh, testmesh::cells> xd, bd;

int driver(flecsi::scheduler & s) {
	std::array<testmesh::ptr, ncases> mptrs;
	std::array cases{std::make_tuple("Chem97ZtZ.mtx", 92),
	                 std::make_tuple("psmigr_3.mtx", 33)};

	static_assert(cases.size() <= ncases);

	UNIT () {
		auto settings =
			read_config("bicgstab.cfg", bicgstab::options("solver"));
		std::size_t i = 0;
		for (const auto & cs : cases) {
			auto & [cname, expected_iters] = cs;
			auto mtx = mat::io::matrix_market<>::read(cname).tocsr();

			auto & msh = init_mesh(s, mtx.rows(), mptrs[i]);

			auto [x, b] = vec::make(msh)(xd, bd);
			b.set_random(0);
			x.set_random(1);

			op::core<csr_op> A(std::move(mtx));
			bicgstab::solver slv(settings, bicgstab::make_work(b));
			auto info = slv(op::ref(A))(b, x);

			EXPECT_EQ(info.status, solve_info::stop_reason::converged_rtol);
			EXPECT_LE(info.iters, expected_iters);

			++i;
		}
	};
}

flecsi::util::unit::driver<driver> drv;

}
