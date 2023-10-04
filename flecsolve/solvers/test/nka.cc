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

testmesh::slot msh;
testmesh::cslot coloring;

const realf::definition<testmesh, testmesh::cells> xd, bd;

struct simple_factory : solver_factory<simple_factory> {
	enum class registry { identity, Dinv };

	void set_solver_type(registry reg) { solver_type = reg; }

	template<class V, class Op>
	void create_parameters(vec::base<V> &, op::base<Op> &) {}

	template<class V, class Op>
	void create(vec::base<V> &, op::base<Op> & A) {
		if (solver_type == registry::Dinv)
			dinv.emplace(A.derived().Dinv());
	}

	template<class V, class D, class R, class Op>
	void solve(const vec::base<D> & x,
	           vec::base<R> & y,
	           vec::base<V> &,
	           op::base<Op> &) {
		if (solver_type == registry::identity)
			op::I.apply(x, y);
		else
			dinv->apply(x, y);
	}

	std::optional<csr_op> dinv;
	registry solver_type;
};

std::istream & operator>>(std::istream & in, simple_factory::registry & reg) {
	std::string tok;
	in >> tok;

	if (tok == "identity")
		reg = simple_factory::registry::identity;
	else if (tok == "Dinv")
		reg = simple_factory::registry::Dinv;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

int nkatest() {
	UNIT () {
		auto mtx = mat::io::matrix_market<>::read("Chem97ZtZ.mtx").tocsr();
		init_mesh(mtx.rows(), msh, coloring);

		csr_op A{std::move(mtx)};
		vec::topo_view x(msh, xd(msh)), b(msh, bd(msh));
		{
			b.set_scalar(1.);
			x.set_scalar(3.);

			cg::settings pre_settings("preconditioner");
			nka::settings nnl_settings("solver");
			read_config("nka.cfg", pre_settings, nnl_settings);

			op::krylov P(op::krylov_parameters(
				pre_settings, cg::topo_work<>::get(b), std::ref(A)));

			op::krylov slv(op::krylov_parameters(nnl_settings,
			                                     nka::topo_work<5>::get(b),
			                                     std::ref(A),
			                                     std::move(P)));
			auto info = slv.apply(b, x);
			EXPECT_EQ(info.iters, 17);
		}
		{
			b.set_scalar(1.);
			x.set_scalar(3.);

			std::size_t iter{0}, inner{0};
			op::krylov_parameters params(
				nka::settings("nnl-solver"),
				nka::topo_work<5>::get(b),
				std::ref(A),
				krylov_factory(factory_union(simple_factory(),
			                                 krylov_factory(simple_factory())),
			                   [&](const auto &, double rnorm) {
								   std::cout << "inner: " << ++inner << " "
											 << rnorm << std::endl;
								   return false;
							   }),
				[&](const auto &, double rnorm) {
					inner = 0;
					std::cout << ++iter << " " << rnorm << std::endl;
					return false;
				});
			read_config("nka-factory.cfg", params);

			op::krylov slv(std::move(params));

			auto info = slv.apply(b, x);
			EXPECT_EQ(info.iters, 3);
		}
	};

	return 0;
}

flecsi::util::unit::driver<nkatest> driver;
}
