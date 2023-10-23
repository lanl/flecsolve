#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>
#include <flecsi/execution.hh>
#include <iomanip>

#include "flecsolve/vectors/seq.hh"
#include "flecsolve/operators/core.hh"
#include "flecsolve/solvers/factory.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/krylov_operator.hh"
#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/matrices/io/matrix_market.hh"
#include "flecsolve/matrices/parcsr.hh"

namespace flecsolve {

using namespace flecsi;

field<double>::definition<mat::parcsr, mat::parcsr::cols> xd;
field<double>::definition<mat::parcsr, mat::parcsr::cols> yd;

namespace mat {

template<class Config>
struct parcsr_data {
	using config = Config;
	parcsr::slot topo;
	parcsr::cslot coloring;
	parcsr::init coloring_input;

	auto spmv_tmp() { return vec::topo_view(topo, spmv_tmp_def(topo)); }

protected:
	typename field<typename Config::scalar>::template definition<parcsr,
	                                                             parcsr::cols>
		spmv_tmp_def;
};

template<class Data>
struct parcsr_ops {
	using data_t = Data;
	using config = typename Data::config;
	using scalar_t = typename config::scalar;
	template<class D, class R>
	static void spmv(const D & x, const data_t & data_c, R & y) {
		auto & data = const_cast<data_t &>(data_c);
		auto tmpv = const_cast<data_t &>(data).spmv_tmp();
		flecsi::execute<spmv_remote>(data.topo, tmpv.data.ref(), x.data.ref());
		flecsi::execute<spmv_local>(data.topo, y.data.ref(), x.data.ref());
		y.add(y, tmpv);
	}

protected:
	static void spmv_remote(
		parcsr::accessor<flecsi::ro> ma,
		typename field<scalar_t>::template accessor<flecsi::wo, flecsi::na> ya,
		typename field<scalar_t>::template accessor<flecsi::na, flecsi::ro>
			xa) {
		vec::seq_view y{ya.span()};
		vec::seq_view x{xa.span()};
		ma.offd().mult(x, y);
	}

	static void spmv_local(
		parcsr::accessor<flecsi::ro> ma,
		typename field<scalar_t>::template accessor<flecsi::wo, flecsi::na> ya,
		typename field<scalar_t>::template accessor<flecsi::ro, flecsi::na>
			xa) {
		vec::seq_view y{ya.span()};
		vec::seq_view x{xa.span()};
		ma.diag().mult(x, y);
	}
};

struct parcsr_config {
	using scalar = double;
	using size = std::size_t;
};
struct parcsr_op
	: flecsolve::mat::sparse<parcsr_data, parcsr_ops, parcsr_config> {
	using base = flecsolve::mat::sparse<parcsr_data, parcsr_ops, parcsr_config>;
	parcsr_op(MPI_Comm comm,
	          const char * fname,
	          Color colors = flecsi::processes())
		: comm_(comm), colors_(colors) {
		flecsi::execute<read_mat, flecsi::mpi>(
			comm, fname, colors, data.coloring_input);
		data.coloring.allocate(data.coloring_input);
		data.topo.allocate(data.coloring.get(), data.coloring_input);
	}

protected:
	static void read_mat(MPI_Comm comm,
	                     const char * fname,
	                     Color colors,
	                     flecsolve::mat::parcsr::init & ci) {
		/*
		  1. distribute csr across colors to each processor
		  2. broadcast global dimensions
		  3. distribute referencers (source pointers) to each processor
		*/
		// read and distribute to colors
		auto [rank, comm_size] = util::mpi::info(comm);
		using scalar = mat::parcsr::scalar;
		using size = mat::parcsr::size;

		mat::io::matrix_market<scalar, size>::definition mdef{fname};
		const util::equal_map pm(colors, comm_size);
		ci.proc_mats = util::mpi::one_to_allv(
			[=, &mdef](int r, int) {
				const util::equal_map rm(mdef.num_rows(), colors);
				std::vector<mat::csr<scalar, size>> proc_mats;

				for (const auto clr : pm[r]) {
					proc_mats.push_back(mdef.matrix(rm[clr]));
				}

				return proc_mats;
			},
			comm);

		std::array<std::size_t, 2> shape;
		if (rank == 0) {
			shape[0] = mdef.num_rows();
			shape[1] = mdef.num_cols();
		}
		MPI_Bcast(shape.data(), 2, util::mpi::type<std::size_t>(), 0, comm);

		ci.comm = comm;
		ci.nrows = shape[0];
		ci.ncols = shape[1];
	}

	MPI_Comm comm_;
	Color colors_;
};
}

int csr_test() {
	UNIT () {
		using namespace flecsolve::mat;

		op::core<parcsr_op> A(MPI_COMM_WORLD, "Chem97ZtZ.mtx");
		auto & topo = A.source().data.topo;
		vec::topo_view x(topo, xd(topo));
		vec::topo_view y(topo, yd(topo));

		y.set_scalar(0.0);
		x.set_scalar(2);

		op::krylov_parameters params{
			cg::settings("solver"), cg::topo_work<>::get(x), std::ref(A)};
		read_config("parcsr.cfg", params);

		auto slv = op::krylov_solver(std::move(params));

		auto info = slv.apply(y, x);
		EXPECT_TRUE(info.iters == 167);
	};
	return 0;
}

flecsi::util::unit::driver<csr_test> driver;
}
