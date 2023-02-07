#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/mesh.hh"
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
		Vec r(x0.data.topo(), resdef(x0.data.topo()));
		A.residual(b, x0, r);
		rnorm0 = r.l2norm().get();
		rnorm_prev = rnorm0;
	}

	bool operator()(const vec::base<Vec> &, double rnorm) {
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
		csr_op A{std::move(matrix)};
		auto Dinv = A.Dinv();

		vec::mesh x(msh, xd(msh)), b(msh, bd(msh));
		b.set_random(0);
		x.set_random(1);

		diagnostic diag(A, x, b, cfact);
		op::krylov_parameters params_norestart(
			gmres::settings("gmres-norestart"),
			gmres::topo_work<>::get(b),
			std::ref(A),
			op::I,
			std::ref(diag));
		op::krylov_parameters params_restart(gmres::settings("gmres-restart"),
		                                     gmres::topo_work<>::get(b),
		                                     std::ref(A));
		read_config("gmres.cfg", params_norestart, params_restart);
		{
			op::krylov slv(std::move(params_norestart));
			auto info = slv.apply(b, x);

			EXPECT_EQ(info.iters, 73);
			EXPECT_FALSE(diag.fail_monotonic);
			EXPECT_FALSE(diag.fail_convergence);

			auto slv_pre = op::rebind(slv, std::ref(A), std::ref(Dinv));
			x.set_random(1);
			auto info_pre = slv_pre.apply(b, x);
			EXPECT_EQ(info_pre.iters, 18);
		}
		{ // test restart
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
