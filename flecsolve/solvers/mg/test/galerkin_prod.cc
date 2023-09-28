#include "flecsi/flog.hh"

#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/matrices/seq.hh"
#include "flecsolve/matrices/parcsr.hh"
#include "flecsolve/solvers/mg/coarsen.hh"
#include <limits>

namespace flecsolve {

using namespace flecsi;

using scalar = double;

using csr = topo::csr<scalar>;
using parcsr = mat::parcsr<scalar>;
field<util::id>::definition<csr, csr::cols> aggt_def;

field<scalar>::definition<csr, csr::cols> xd, yd, y1d, zd, wd;

namespace {

constexpr float tol = 1e-9;

void init(csr::accessor<ro> m, field<scalar>::accessor<wo, na> x) {
	for (auto dof : m.dofs<csr::cols>()) {
		x[dof] = static_cast<scalar>(m.global_id(dof)) / m.meta().nrows;
	}
}

void explicit_interp(csr::accessor<ro> mf,
                     csr::accessor<ro> mc,
                     field<util::id>::accessor<ro, na> aggt,
                     field<scalar>::accessor<ro, na> xa,
                     field<scalar>::accessor<wo, na> za) {
	mat::csr<scalar> P(mf.meta().rows.size(), mc.meta().rows.size());
	P.resize(P.rows());
	auto [rowptr, colind, values] = P.rep();

	std::size_t off = mc.meta().rows.beg;
	for (std::size_t i{0}; i < P.rows(); ++i) {
		rowptr[i + 1] = i + 1;
		if (aggt[i] == std::numeric_limits<util::id>::max()) {
			values[i] = 0;
			colind[i] = 0;
		}
		else {
			values[i] = 1;
			colind[i] = aggt[i] - off;
		}
	}

	vec::seq_view z{za.span()};
	vec::seq_view x{xa.span()};

	P.apply(x, z);
}

void explicit_restrict(csr::accessor<ro> mf,
                       csr::accessor<ro> mc,
                       field<util::id>::accessor<ro, na> aggt,
                       field<scalar>::accessor<ro, na> wa,
                       field<scalar>::accessor<wo, na> ya) {
	auto R = [&]() {
		mat::coo<scalar> R(mc.meta().rows.size(), mf.meta().rows.size());
		R.resize(R.cols());

		auto & I = R.data.I;
		auto & J = R.data.J;
		auto & V = R.data.V;
		std::size_t off = mc.meta().rows.beg;
		for (std::size_t i{0}; i < R.cols(); ++i) {
			J[i] = i;
			if (aggt[i] == std::numeric_limits<util::id>::max()) {
				V[i] = 0;
				I[i] = 0;
			}
			else {
				V[i] = 1;
				I[i] = aggt[i] - off;
			}
		}
		return R.tocsr();
	}();

	vec::seq_view w{wa.span()};
	vec::seq_view y{ya.span()};

	R.apply(w, y);
}

int coarsentest() {
	UNIT () {
		parcsr A{parcsr::parameters{
			MPI_COMM_WORLD, flecsi::processes(), "nos7.mtx"}};
		auto Ac = mg::ua::coarsen(A, aggt_def(A.data.topo()));

		auto x = Ac.vec(xd);
		auto y = Ac.vec(yd);

		execute<init>(x.data.topo(), x.data.ref());
		Ac.apply(x, y);

		auto z = A.vec(zd);
		execute<explicit_interp>(A.data.topo(),
		                         Ac.data.topo(),
		                         aggt_def(A.data.topo()),
		                         x.data.ref(),
		                         z.data.ref());

		auto w = A.vec(wd);
		A.apply(z, w);

		auto y1 = Ac.vec(y1d);
		execute<explicit_restrict>(A.data.topo(),
		                           Ac.data.topo(),
		                           aggt_def(A.data.topo()),
		                           w.data.ref(),
		                           y1.data.ref());

		y.subtract(y, y1);
		auto diff = y.inf_norm().get();

		EXPECT_LT(diff, tol);
	};
	return 0;
}

flecsi::util::unit::driver<coarsentest> driver;
}

}
