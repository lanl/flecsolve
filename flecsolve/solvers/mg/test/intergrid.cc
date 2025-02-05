#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>
#include <limits>

#include "flecsolve/matrices/parcsr.hh"
#include "flecsolve/solvers/mg/coarsen.hh"
#include "flecsolve/solvers/mg/intergrid.hh"

namespace flecsolve {
namespace {

using namespace flecsi;
using scalar = double;
using parcsr = mat::parcsr<scalar>;
using csr_topo = parcsr::topo_t;

field<util::id>::definition<csr_topo, csr_topo::cols> aggt_def;
field<scalar>::definition<csr_topo, csr_topo::cols> xd, yd;

void init(csr_topo::accessor<ro> m, field<scalar>::accessor<wo, na> x) {
	for (auto dof : m.dofs<csr_topo::cols>()) {
		x[dof] = m.global_id(dof);
	}
}

int check_restrict(csr_topo::accessor<ro> fine,
                   csr_topo::accessor<ro> coarse,
                   field<util::id>::accessor<ro, na> aggt,
                   field<scalar>::accessor<ro, na> x) {
	UNIT () {
		std::vector<scalar> bless;
		bless.resize(coarse.dofs<csr_topo::cols>().size());
		for (auto dof : fine.dofs<csr_topo::cols>()) {
			if (aggt[dof] != std::numeric_limits<util::id>::max())
				bless[aggt[dof] - coarse.meta().rows.beg] +=
					fine.global_id(dof);
		}
		for (auto dof : coarse.dofs<csr_topo::cols>()) {
			EXPECT_EQ(bless[dof], x[dof]);
		}
	};
}

int check_interp(csr_topo::accessor<ro> fine,
                 csr_topo::accessor<ro> coarse,
                 field<util::id>::accessor<ro, na> aggt,
                 field<scalar>::accessor<ro, na> x) {

	UNIT () {
		for (auto dof : fine.dofs<csr_topo::cols>()) {
			if (aggt[dof] != std::numeric_limits<util::id>::max()) {
				EXPECT_EQ(x[dof],
				          coarse.global_id(flecsi::topo::id<csr_topo::cols>(
							  aggt[dof] - coarse.meta().cols.beg)));
			}
		}
	};
}

int intergridtest() {
	UNIT () {
		op::core<parcsr> A(MPI_COMM_WORLD, "nos7.mtx");

		auto & topof = A.data.topo();
		auto aggt_ref = aggt_def(topof);
		auto Ac = op::make(
			mg::ua::coarsen<scalar, std::size_t>(A, aggt_ref));
		execute<init>(topof, xd(topof));

		mg::ua::intergrid_params<scalar, std::size_t> params{aggt_ref};

		mg::ua::prolong<scalar, std::size_t> P(params);
		mg::ua::restrict<scalar, std::size_t> R(params);

		auto x = A.vec(xd);
		auto y = Ac.vec(yd);

		R.apply(x, y);

		auto & topoc = Ac.data.topo();

		EXPECT_EQ(test<check_restrict>(topof, topoc, aggt_ref, y.data.ref()),
		          0);

		execute<init>(topoc, y.data.ref());

		P.apply(y, x);

		EXPECT_EQ(test<check_interp>(topof, topoc, aggt_ref, x.data.ref()), 0);
	};
	return 0;
}

flecsi::util::unit::driver<intergridtest> driver;
}
}
