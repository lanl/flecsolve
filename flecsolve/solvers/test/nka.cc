#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/util/config.hh"
#include "flecsolve/solvers/krylov_operator.hh"
#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/nka.hh"
#include "flecsolve/solvers/factory.hh"
#include "flecsolve/matrices/io/matrix_market.hh"
#include "flecsolve/util/test/mesh.hh"

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
	static auto make(const settings &, Op & A) {
		if constexpr (V == simple_target::identity) {
			return op::I;
		}
		else if constexpr (V == simple_target::Dinv) {
			return A.Dinv();
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

		auto mtx = mat::io::matrix_market<>::read("Chem97ZtZ.mtx").tocsr();
		init_mesh(mtx.rows(), msh);

		auto A = op::make_shared<csr_op>(std::move(mtx));
		auto [x, b] = vec::make(msh)(xd, bd);
		{
			b.set_scalar(1.);
			x.set_scalar(3.);

			auto [pre_settings, nnl_settings] =
				read_config("nka.cfg",
			                cg::options("preconditioner"),
			                nka::options("solver"));

			auto P = op::make(op::krylov(op::krylov_parameters(
				pre_settings, cg::topo_work<>::get(b), A)));

			op::krylov slv(op::krylov_parameters(
				nnl_settings, nka::topo_work<5>::get(b), A, std::move(P)));
			auto info = slv.apply(b, x);
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

			op::krylov_parameters params(
				nnl_settings,
				nka::topo_work<5>::get(b),
				A,
				krylov_factory::make(
					lin_settings,
					b,
					A,
					simple_factory::make(precond_settings, *A)));

			op::krylov slv(std::move(params));

			auto info = slv.apply(b, x);
			EXPECT_EQ(info.iters, 3);
		}
	};

	return 0;
}

flecsi::util::unit::driver<nkatest> driver;
}
