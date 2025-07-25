/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
#ifndef FLECSOLVE_MATRICES_PARCSR_H
#define FLECSOLVE_MATRICES_PARCSR_H

#include <flecsi/flog.hh>
#include <flecsi/execution.hh>

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/vectors/core.hh"
#include "flecsolve/topo/csr.hh"
#include "flecsolve/matrices/io/matrix_market.hh"

namespace flecsolve::mat {

template<class Config>
struct parcsr_data {
	using config = Config;
	using topo_t =
		typename topo::csr<typename config::scalar, typename config::size>;
	auto & topo() const { return *topo_ptr; }
	typename topo_t::init coloring_input;

	auto spmv_tmp() { return vec::make(spmv_tmp_def(topo())); }
	auto nrows() const { return coloring_input.nrows; }

	void allocate(flecsi::scheduler & s) {
		s.allocate(topo_ptr, typename topo_t::mpi_coloring(s, coloring_input), coloring_input);
	}

protected:
	typename topo_t::ptr topo_ptr;
	inline static typename flecsi::field<
		typename Config::scalar>::template definition<topo_t, topo_t::cols>
		spmv_tmp_def;
};

template<class Data>
struct parcsr_ops {
	using data_t = Data;
	using config = typename Data::config;
	using scalar_t = typename config::scalar;
	using topo_t = typename Data::topo_t;
	template<flecsi::privilege p>
	using topo_acc = typename topo_t::template accessor<p>;

	template<class D, class R>
	static void spmv(const D & x, const data_t & data_c, R & y) {
		auto & data = const_cast<data_t &>(data_c);
		auto tmpv = const_cast<data_t &>(data).spmv_tmp();
		flecsi::execute<spmv_remote>(
			data.topo(), tmpv.data.ref(), x.data.ref());
		flecsi::execute<spmv_local>(data.topo(), y.data.ref(), x.data.ref());
		y.add(y, tmpv);
	}

protected:
	static void spmv_remote(
		topo_acc<flecsi::ro> ma,
		typename flecsi::field<scalar_t>::template accessor<flecsi::wo,
	                                                        flecsi::na> ya,
		typename flecsi::field<scalar_t>::template accessor<flecsi::na,
	                                                        flecsi::ro> xa) {
		vec::seq_view y{ya.span()};
		vec::seq_view x{xa.span()};
		ma.offd().mult(x, y);
	}

	static void spmv_local(
		topo_acc<flecsi::ro> ma,
		typename flecsi::field<scalar_t>::template accessor<flecsi::wo,
	                                                        flecsi::na> ya,
		typename flecsi::field<scalar_t>::template accessor<flecsi::ro,
	                                                        flecsi::na> xa) {
		vec::seq_view y{ya.span()};
		vec::seq_view x{xa.span()};
		ma.diag().mult(x, y);
	}
};

template<class scalar_t, class size_t>
struct parcsr_config {
	using scalar = scalar_t;
	using size = size_t;
};

template<class scalar, class size = std::size_t>
struct parcsr : flecsolve::mat::sparse<parcsr_data,
                                       parcsr_ops,
                                       parcsr_config<scalar, size>> {
	using base = flecsolve::mat::
		sparse<parcsr_data, parcsr_ops, parcsr_config<scalar, size>>;
	using data_t = typename base::data_t;
	using topo_t = typename data_t::topo_t;
	using base::data;
	using scalar_type = scalar;
	using size_type = size;

	parcsr(flecsi::scheduler & s,
	       MPI_Comm comm,
	       const char * fname)
		: comm_(comm), colors_(s.runtime().processes()) {
		flecsi::execute<read_mat, flecsi::mpi>(
			comm, fname, colors_, data.coloring_input);
		data.allocate(s);
	}

	parcsr(flecsi::scheduler & s, typename topo_t::init && init) {
		data.coloring_input = std::move(init);
		data.allocate(s);
	}

	template<typename topo::csr<scalar, size>::index_space S>
	auto vec(typename flecsi::field<
	         scalar>::template definition<topo::csr<scalar, size>,  S> & def) {
		return vec::make(def(data.topo()));
	}

protected:
	static void read_mat(MPI_Comm comm,
	                     const char * fname,
	                     flecsi::Color colors,
	                     typename topo_t::init & ci) {
		/*
		  1. distribute csr across colors to each processor
		  2. broadcast global dimensions
		  3. distribute referencers (source pointers) to each processor
		*/
		// read and distribute to colors
		auto [rank, comm_size] = flecsi::util::mpi::info(comm);

		typename mat::io::matrix_market<scalar, size>::definition mdef{fname};
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
		MPI_Bcast(
			shape.data(), 2, flecsi::util::mpi::type<std::size_t>(), 0, comm);

		ci.comm = comm;
		ci.nrows = shape[0];
		ci.ncols = shape[1];
		ci.row_part.set_block_map(ci.nrows, colors);
		ci.col_part.set_block_map(ci.ncols, colors);
		ci.proc_part.set_block_map(colors, comm_size);
	}

	MPI_Comm comm_;
	flecsi::Color colors_;
};
}

#endif
