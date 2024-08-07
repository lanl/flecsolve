#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/util/config.hh"
#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/nka.hh"
#include "flecsolve/matrices/io/matrix_market.hh"
#include "flecsolve/util/test/mesh.hh"
#include "flecsolve/solvers/factory.hh"

namespace flecsolve {

const realf::definition<testmesh, testmesh::cells> xd, bd;

enum class simple_target { identity, Dinv };

std::istream & operator>>(std::istream & in, simple_target & reg) {
	std::string tok;
	in >> tok;

	if (tok == "identity")
		reg = simple_target::identity;
	else if (tok == "Dinv")
		reg = simple_target::Dinv;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

template<simple_target V>
struct simple_registry {
	using settings = null_settings<V>;
	using options = null_options<V>;
	template<class Op>
	static auto make(const settings &, op::handle<Op> & A) {
		if constexpr (V == simple_target::identity) {
			return op::I.get();
		}
		else if constexpr (V == simple_target::Dinv) {
			return A.get().Dinv();
		}
	}
};

struct simple_policy {
	using target = simple_target;
	using targets = includes<target::identity, target::Dinv>;
	template<target V>
	using registry = simple_registry<V>;
};
using simple_factory = op::factory<simple_policy>;

int nkatest() {
	UNIT () {
		testmesh::slot msh;

		auto A = op::make_shared<csr_op>(
			mat::io::matrix_market<>::read("Chem97ZtZ.mtx").tocsr());

		init_mesh(A.get().rows(), msh);

		auto [x, b] = vec::make(msh)(xd, bd);
		{
			b.set_scalar(1.);
			x.set_scalar(3.);

			auto [pre_settings, nnl_settings] =
				read_config("nka.cfg",
			                cg::options("preconditioner"),
			                nka::options("solver"));

			auto P = cg::solver(pre_settings, cg::make_work(b))(A);
			auto slv = nka::solver(nnl_settings,
			                       nka::make_work(nka::dim_bound<5>, b))(A, op::ref(P));
			auto info = slv(b, x);
			EXPECT_EQ(info.iters, 17);
		}
		{
			b.set_scalar(1.);
			x.set_scalar(3.);

			auto [nnl_settings, lin_settings, precond_settings] =
				read_config("nka-factory.cfg",
			                nka::options("nnl-solver"),
			                krylov_factory::options("linear-solver"),
			                simple_factory::options("inner"));

			auto slv = nka::solver(
				nnl_settings,
				nka::make_work(nka::dim_bound<10>, b))(
					A,
					krylov_factory::make_shared(
						lin_settings, b, A,
						simple_factory::make_shared(
							precond_settings, A)));
			auto info = slv(b, x);
			EXPECT_EQ(info.iters, 3);
		}
	};

	return 0;
}

flecsi::util::unit::driver<nkatest> driver;
}
