#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/solvers/gmres.hh"
#include "flecsolve/util/config.hh"
#include "flecsolve/matrices/io/matrix_market.hh"
#include "flecsolve/util/test/mesh.hh"

namespace flecsolve {

static constexpr std::size_t ncases = 2;
std::array<testmesh::slot, ncases> mshs;
std::array<testmesh::cslot, ncases> colorings;

const realf::definition<testmesh, testmesh::cells> xd, bd;

template<class Op, class Vec>
struct diagnostic {
	diagnostic(const Op & A, const Vec & x0, const Vec & b, double cfact)
		: iter(0), cfact(cfact), fail_monotonic(false),
		  fail_convergence(false) {
		auto r = vec::make(x0.data.topo(), resdef(x0.data.topo()));
		A.residual(b, x0, r);
		rnorm0 = r.l2norm().get();
		rnorm_prev = rnorm0;
	}

	template<class V, std::enable_if_t<is_vector_v<V>, bool> = true>
	bool operator()(const V &, double rnorm) {
		float n = ++iter;
		auto bnd = std::pow(cfact, n / 2) * rnorm0;

		fail_monotonic = fail_monotonic || (rnorm > rnorm_prev);
		fail_convergence = fail_convergence || (rnorm > bnd);

		rnorm_prev = rnorm;

		return false;
	}

	std::size_t iter;
	double rnorm0, rnorm_prev;
	double cfact;

	bool fail_monotonic;
	bool fail_convergence;

	static inline const realf::definition<testmesh, testmesh::cells> resdef;
};

int gmres_test() {
	UNIT () {
		auto matrix = mat::io::matrix_market<>::read("Chem97ZtZ.mtx").tocsr();

		double cond = 2.472189e+02;
		double cfact = (cond * cond - 1) / (cond * cond);

		auto & msh = mshs[0];
		init_mesh(matrix.rows(), msh, colorings[0]);
		op::core<csr_op, op::shared_storage> A(std::move(matrix));
		op::core<csr_op, op::shared_storage> Dinv(A.source().Dinv());

		auto [x, b] = vec::make(msh)(xd, bd);
		b.set_random(0);
		x.set_random(1);

		diagnostic diag(A, x, b, cfact);
		auto [settings_norestart, settings_restart] =
			read_config("gmres.cfg",
		                gmres::options("gmres-norestart"),
		                gmres::options("gmres-restart"));
		{
			op::krylov slv(op::krylov_parameters(settings_norestart,
			                                     gmres::topo_work<>::get(b),
			                                     A,
			                                     op::I,
			                                     std::ref(diag)));
			auto info = slv.apply(b, x);

			EXPECT_EQ(info.iters, 73);
			EXPECT_FALSE(diag.fail_monotonic);
			EXPECT_FALSE(diag.fail_convergence);

			auto slv_pre = op::rebind(slv, A, Dinv);
			x.set_random(1);
			auto info_pre = slv_pre.apply(b, x);
			EXPECT_EQ(info_pre.iters, 18);
		}
		{ // test restart
			op::krylov_parameters params_restart(
				settings_restart, gmres::topo_work<>::get(b), A);
			b.set_random(0);
			x.set_random(1);

			op::krylov slv(params_restart);
			auto info_restart = slv.apply(b, x);

			x.set_random(1);

			params_restart.solver_settings.maxiter = 50;
			slv.reset(params_restart.solver_settings);
			slv.apply(b, x);

			params_restart.solver_settings.maxiter = 100;
			params_restart.solver_settings.max_krylov_dim = 100;
			params_restart.solver_settings.restart = false;
			op::krylov slv1(params_restart);

			auto info = slv1.apply(b, x);
			EXPECT_EQ(50 + info.iters, info_restart.iters);
			EXPECT_EQ(info.res_norm_final, info_restart.res_norm_final);
		}
	};

	return 0;
}

flecsi::util::unit::driver<gmres_test> driver;

}
