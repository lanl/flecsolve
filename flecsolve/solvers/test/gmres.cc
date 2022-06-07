#include <cmath>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/solvers/gmres.hh"

#include "csr_utils.hh"

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
		Vec r(x0.data.topo, resdef(x0.data.topo));
		A.residual(b, x0, r);
		rnorm0 = r.l2norm().get();
		rnorm_prev = rnorm0;
	}

	bool operator()(const Vec &, double rnorm) {
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

static csr<> get_idiag(const csr<> & in) {
	csr<> out{in.nrows, in.nrows};

	for (std::size_t i = 0; i < in.nrows; i++) {
		for (std::size_t off = in.rowptr[i]; off < in.rowptr[i + 1]; off++) {
			if (in.colind[off] == i) {
				out.values[i] = 1.0 / in.values[off];
				out.colind[i] = i;
			}
		}
		out.rowptr[i + 1] = i + 1;
	}

	return out;
}

int gmres_test() {
	UNIT () {
		auto mat = read_mm("Chem97ZtZ.mtx");
		auto idiag = get_idiag(mat);

		double cond = 2.472189e+02;
		double cfact = (cond * cond - 1) / (cond * cond);

		auto & msh = mshs[0];
		init_mesh(mat.nrows, msh, colorings[0]);
		csr_op A{std::move(mat)};
		csr_op Dinv{std::move(idiag)};

		vec::mesh x(msh, xd(msh)), b(msh, bd(msh));

		{
			b.set_random(0);
			x.set_random(1);

			diagnostic diag(A, x, b, cfact);
			krylov_params params(gmres::settings{100, 1e-4},
			                     gmres::topo_work<>::get(b),
			                     A,
			                     op::I,
			                     diag);
			auto slv = op::create(std::move(params));

			auto info = slv.apply(b, x);

			EXPECT_EQ(info.iters, 73);
			EXPECT_FALSE(diag.fail_monotonic);
			EXPECT_FALSE(diag.fail_convergence);

			auto slv_pre = slv.rebind(A, Dinv);
			x.set_random(1);
			auto info_pre = slv_pre.apply(b, x);
			EXPECT_EQ(info_pre.iters, 18);
		}
		{ // test restart
			b.set_random(0);
			x.set_random(1);

			krylov_params params(
				gmres::settings{100, 50, 1e-4}, gmres::topo_work<>::get(b), A);

			auto slv = op::create(params);
			auto info_restart = slv.apply(b, x);

			x.set_random(1);

			params.solver_settings.maxiter = 50;
			slv.solver.reset(params.solver_settings);
			slv.apply(b, x);

			params.solver_settings.maxiter = 100;
			params.solver_settings.max_krylov_dim = 100;
			params.solver_settings.restart = false;
			auto slv1 = op::create(params);

			auto info = slv1.apply(b, x);
			EXPECT_EQ(50 + info.iters, info_restart.iters);
			EXPECT_EQ(info.res_norm_final, info_restart.res_norm_final);
		}
	};

	return 0;
}

flecsi::unit::driver<gmres_test> driver;

}
