#include <array>
#include <cassert>

#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsi-linalg/discrete_operators/boundary/dirichlet.hh"
#include "flecsi-linalg/discrete_operators/expressions/operator_expression.hh"
#include "flecsi-linalg/operators/cg.hh"
#include "flecsi-linalg/operators/solver_settings.hh"
#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/vectors/multi.hh"
#include "flecsi-linalg/vectors/variable.hh"

#include "flecsi-linalg/discrete_operators/specializations/operator_mesh.hh"

using namespace flecsi;

namespace flecsi::linalg {
using msh = discrete_operators::operator_mesh;

constexpr std::size_t NX = 8;
constexpr std::size_t NY = 8;

msh::slot m;
msh::cslot coloring;

const field<double>::definition<msh, msh::cells> xd, yd;
const field<double>::definition<msh, msh::cells> rhsxd, rhsyd;

enum class bndvar { v1 = 1, v2 };

void init_mesh() {
	std::vector<std::size_t> extents{{NX, NY}};
	auto colors = msh::distribute(flecsi::processes(), extents);
	coloring.allocate(colors, extents);

	msh::grect geometry;
	geometry[0][0] = 0.0;
	geometry[0][1] = 1.0;
	geometry[1] = geometry[0];

	m.allocate(coloring.get(), geometry);
}

int multivectorop_test() {

	init_mesh();

	vec::mesh xvec(variable<bndvar::v1>, m, xd(m));
	vec::mesh yvec(variable<bndvar::v2>, m, yd(m));

	vec::multi X(xvec, yvec);

	vec::mesh rhsvx(variable<bndvar::v1>, m, rhsxd(m));
	vec::mesh rhsvy(variable<bndvar::v2>, m, rhsyd(m));
	vec::multi RHS(rhsvx, rhsvy);

	X.set_scalar(1.0);
	RHS.set_scalar(0.0);

	auto bndry_xlow = discrete_operators::make_operator<
		discrete_operators::
			dirchilet<bndvar::v1, msh, msh::x_axis, msh::boundary_low>>(0.0);
	auto A = discrete_operators::op_expr(bndry_xlow);

	UNIT () {
		krylov_params params(cg::settings{100, 1e-9, 1e-9},
		                     cg::topo_work<>::get(RHS),
		                     std::move(A));
		EXPECT_TRUE(params.work[0].getvar<bndvar::v1>().data.fid() < 1000);
		auto slv = linalg::op::create(std::move(params));
		EXPECT_TRUE(params.work[0].getvar<bndvar::v1>().data.fid() > 1000);
		auto info = slv.apply(RHS, X);
	};
}

unit::driver<multivectorop_test> driver;

} // namespace flecsi::linalg