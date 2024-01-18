#ifndef FLECSOLVE_MATRICES_PARCSR_H
#define FLECSOLVE_MATRICES_PARCSR_H

#include <flecsi/flog.hh>
#include <flecsi/execution.hh>

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/vectors/base.hh"
#include "flecsolve/topo/csr.hh"
#include "flecsolve/matrices/io/matrix_market.hh"

namespace flecsolve::mat {

template<class Config>
struct parcsr_data {
	using config = Config;
	using topo_t = typename topo::csr<typename config::scalar, typename config::size>;
	typename topo_t::slot & topo() {
		if (!topo_slot)
			topo_slot = std::make_unique<typename topo_t::slot>();
		return *topo_slot;
	}
	typename topo_t::slot & topo() const {
		return *topo_slot;
	}
	typename topo_t::cslot coloring;
	typename topo_t::init coloring_input;

	auto spmv_tmp() { return vec::make(topo(), spmv_tmp_def(topo())); }

protected:
	std::unique_ptr<typename topo_t::slot> topo_slot;
	inline static typename flecsi::field<typename Config::scalar>::template definition<topo_t,
	                                                             topo_t::cols>
		spmv_tmp_def;
};

template<class Data>
struct parcsr_ops {
	using data_t = Data;
	using config = typename Data::config;
	using scalar_t = typename config::scalar;
	using topo_t = typename Data::topo_t;
	template<flecsi::partition_privilege_t p>
	using topo_acc = typename topo_t::template accessor<p>;

	template<class D, class R>
	static void spmv(const D & x, const data_t & data_c, R & y) {
		auto & data = const_cast<data_t &>(data_c);
		auto tmpv = const_cast<data_t &>(data).spmv_tmp();
		flecsi::execute<spmv_remote>(data.topo(), tmpv.data.ref(), x.data.ref());
		flecsi::execute<spmv_local>(data.topo(), y.data.ref(), x.data.ref());
		y.add(y, tmpv);
	}

protected:
	static void spmv_remote(
		topo_acc<flecsi::ro> ma,
		typename flecsi::field<scalar_t>::template accessor<flecsi::wo, flecsi::na> ya,
		typename flecsi::field<scalar_t>::template accessor<flecsi::na, flecsi::ro>
			xa) {
		vec::seq_view y{ya.span()};
		vec::seq_view x{xa.span()};
		ma.offd().mult(x, y);
	}

	static void spmv_local(
		topo_acc<flecsi::ro> ma,
		typename flecsi::field<scalar_t>::template accessor<flecsi::wo, flecsi::na> ya,
		typename flecsi::field<scalar_t>::template accessor<flecsi::ro, flecsi::na>
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
	using topo_t = typename data_t::topo_t;
	parcsr_op(MPI_Comm comm,
	          const char * fname,
	          flecsi::Color colors = flecsi::processes())
		: comm_(comm), colors_(colors) {
		flecsi::execute<read_mat, flecsi::mpi>(
			comm, fname, colors, data.coloring_input);
		data.coloring.allocate(data.coloring_input);
		data.topo().allocate(data.coloring.get(), data.coloring_input);
	}

	explicit parcsr_op(topo_t::init && init) {
		data.coloring_input = std::move(init);
		data.coloring.allocate(data.coloring_input);
		data.topo().allocate(data.coloring.get(), data.coloring_input);
	}

	template<typename topo::csr<scalar, size>::index_space S>
	auto vec(typename flecsi::field<
			 scalar>::template definition<topo::csr<scalar, size>, S> & def) {
		return vec::make(data.topo(), def(data.topo()));
	}

protected:
	static void read_mat(MPI_Comm comm,
	                     const char * fname,
	                     flecsi::Color colors,
	                     topo_t::init & ci) {
		/*
		  1. distribute csr across colors to each processor
		  2. broadcast global dimensions
		  3. distribute referencers (source pointers) to each processor
		*/
		// read and distribute to colors
		auto [rank, comm_size] = flecsi::util::mpi::info(comm);

		mat::io::matrix_market<scalar, size>::definition mdef{fname};
		const flecsi::util::equal_map pm(colors, comm_size);
		ci.proc_mats = flecsi::util::mpi::one_to_allv(
			[=, &mdef](int r, int) {
				const flecsi::util::equal_map rm(mdef.num_rows(), colors);
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
		MPI_Bcast(shape.data(), 2, flecsi::util::mpi::type<std::size_t>(), 0, comm);

		ci.comm = comm;
		ci.nrows = shape[0];
		ci.ncols = shape[1];
		ci.row_part.set_block_map(ci.nrows, colors);
		ci.col_part.set_block_map(ci.ncols, colors);
	}

	MPI_Comm comm_;
	flecsi::Color colors_;
};
}

#endif
