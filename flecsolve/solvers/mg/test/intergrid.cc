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
using csr = topo::csr<scalar>;
using parcsr = mat::parcsr<scalar>;

field<util::id>::definition<csr, csr::cols> aggt_def;
field<scalar>::definition<csr, csr::cols> xd, yd;

void init(csr::accessor<ro> m, field<scalar>::accessor<wo, na> x) {
	for (auto dof : m.dofs<csr::cols>()) {
		x[dof] = m.global_id(dof);
	}
}

int check_restrict(csr::accessor<ro> fine,
                   csr::accessor<ro> coarse,
                   field<util::id>::accessor<ro, na> aggt,
                   field<scalar>::accessor<ro, na> x) {
	UNIT () {
		std::vector<scalar> bless;
		bless.resize(coarse.dofs<csr::cols>().size());
		for (auto dof : fine.dofs<csr::cols>()) {
			if (aggt[dof] != std::numeric_limits<util::id>::max())
				bless[aggt[dof] - coarse.meta().rows.beg] +=
					fine.global_id(dof);
		}
		for (auto dof : coarse.dofs<csr::cols>()) {
			EXPECT_EQ(bless[dof], x[dof]);
		}
	};
}

int check_interp(csr::accessor<ro> fine,
                 csr::accessor<ro> coarse,
                 field<util::id>::accessor<ro, na> aggt,
                 field<scalar>::accessor<ro, na> x) {

	UNIT () {
		for (auto dof : fine.dofs<csr::cols>()) {
			if (aggt[dof] != std::numeric_limits<util::id>::max()) {
				EXPECT_EQ(x[dof],
				          coarse.global_id(flecsi::topo::id<csr::cols>(
							  aggt[dof] - coarse.meta().cols.beg)));
			}
		}
	};
}

int intergridtest() {
	UNIT () {
		parcsr A{parcsr::parameters{
			MPI_COMM_WORLD, flecsi::processes(), "nos7.mtx"}};
		auto Ac = mg::ua::coarsen(A, aggt_def(A.data.topo()));
		execute<init>(A.data.topo(), xd(A.data.topo()));

		mg::ua::intergrid_params<scalar, std::size_t> params{
			aggt_def(A.data.topo())};
		mg::ua::prolong<scalar, std::size_t> P(params);
		mg::ua::restrict<scalar, std::size_t> R(params);

		auto x = A.vec(xd);
		auto y = Ac.vec(yd);

		R.apply(x, y);

		EXPECT_EQ(test<check_restrict>(A.data.topo(),
		                               Ac.data.topo(),
		                               aggt_def(A.data.topo()),
		                               y.data.ref()),
		          0);

		execute<init>(Ac.data.topo(), y.data.ref());

		P.apply(y, x);

		EXPECT_EQ(test<check_interp>(A.data.topo(),
		                             Ac.data.topo(),
		                             aggt_def(A.data.topo()),
		                             x.data.ref()),
		          0);
	};
	return 0;
}

flecsi::util::unit::driver<intergridtest> driver;
}
}
