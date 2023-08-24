#ifndef FLECSOLVE_MATRICES_PARCSR_H
#define FLECSOLVE_MATRICES_PARCSR_H

#include "flecsolve/topo/csr.hh"
#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/matrices/io/matrix_market.hh"

namespace flecsolve {

namespace mat {
template<class scalar, class size>
struct parcsr_params {
	using topo_init_t = typename topo::csr<scalar, size>::init;

	parcsr_params(MPI_Comm comm, flecsi::Color colors, std::string fname) {
		flecsi::execute<dist_read, flecsi::mpi>(
			comm, fname.c_str(), colors, topo_init);
	}

	parcsr_params(topo_init_t init) : topo_init(std::move(init)) {}

	topo_init_t topo_init;

private:
	static void dist_read(MPI_Comm comm,
	                      const char * fname,
	                      flecsi::Color colors,
	                      typename topo::csr<scalar, size>::init & ci) {
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
	}
};
template<class scalar, class size>
struct parcsr;
}

namespace op {
template<class scalar, class size>
struct traits<mat::parcsr<scalar, size>> {
	static constexpr auto input_var = variable<anon_var::anonymous>;
	static constexpr auto output_var = variable<anon_var::anonymous>;
	using parameters = mat::parcsr_params<scalar, size>;
};
}
}

namespace flecsolve::mat {

template<class scalar, class size>
struct traits<parcsr<scalar, size>> {
	using scalar_t = scalar;
	using size_t = size;
	using topo_t = topo::csr<scalar, size>;
	struct data_t {
		typename topo_t::slot & topo() {
			if (!topo_slot)
				topo_slot = std::make_unique<typename topo_t::slot>();
			return *topo_slot;
		}
		typename topo_t::slot & topo() const { return *topo_slot; }
		typename topo_t::cslot coloring;

		auto spmv_tmp() { return vec::mesh(topo(), spmv_tmp_def(topo())); }

	protected:
		std::unique_ptr<typename topo_t::slot> topo_slot;
		inline static
			typename flecsi::field<scalar_t>::template definition<topo_t,
		                                                          topo_t::cols>
				spmv_tmp_def;
	};

	struct ops_t {
		template<class D, class R>
		void spmv(const D & x, const data_t & data_c, R & y) const {
			auto & data = const_cast<data_t &>(data_c);
			auto tmpv = const_cast<data_t &>(data).spmv_tmp();
			flecsi::execute<spmv_remote>(
				data.topo(), tmpv.data.ref(), x.data.ref());
			flecsi::execute<spmv_local>(
				data.topo(), y.data.ref(), x.data.ref());
			y.add(y, tmpv);
		}

	protected:
		static void spmv_remote(
			typename topo_t::template accessor<flecsi::ro> ma,
			typename flecsi::field<scalar_t>::template accessor<flecsi::wo,
		                                                        flecsi::na> ya,
			typename flecsi::field<
				scalar_t>::template accessor<flecsi::na, flecsi::ro> xa) {
			vec::seq_view y{ya.span()};
			vec::seq_view x{xa.span()};
			ma.offd().mult(x, y);
		}

		static void spmv_local(
			typename topo_t::template accessor<flecsi::ro> ma,
			typename flecsi::field<scalar_t>::template accessor<flecsi::wo,
		                                                        flecsi::na> ya,
			typename flecsi::field<
				scalar_t>::template accessor<flecsi::ro, flecsi::na> xa) {
			vec::seq_view y{ya.span()};
			vec::seq_view x{xa.span()};
			ma.diag().mult(x, y);
		}
	};
};

template<class scalar, class size = std::size_t>
struct parcsr : sparse<parcsr<scalar, size>> {
	using base = sparse<parcsr<scalar, size>>;
	using base::data;
	using base::params;
	using parameters = typename base::params_type;

	parcsr(parameters p) : base(std::move(p)) {
		data.coloring.allocate(params.topo_init);
		data.topo().allocate(data.coloring.get(), params.topo_init);
	}

	template<typename topo::csr<scalar, size>::index_space S>
	auto vec(typename flecsi::field<
			 scalar>::template definition<topo::csr<scalar, size>, S> & def) {
		return vec::mesh(data.topo(), def(data.topo()));
	}
};

}

#endif
