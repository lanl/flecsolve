#ifndef FLECSOLVE_MATRICES_PAR_H
#define FLECSOLVE_MATRICES_PAR_H

#include "flecsi/topo/unstructured/interface.hh"
// for some reason this is needed
#include "flecsi/data/coloring.hh"

#include "flecsolve/vectors/seq.hh"
#include "flecsolve/matrices/io/matrix_market.hh"
#include "flecsolve/matrices/coloring_utils.hh"

namespace flecsolve::mat {

enum class storage_format { csr };
template<storage_format format>
struct storage {};


struct par :
		flecsi::topo::help,
		flecsi::topo::specialization<flecsi::topo::unstructured, par>
{
	using coloring = flecsi::topo::unstructured_base::coloring;

	template<auto>
	static constexpr flecsi::PrivilegeCount privilege_count = 3;

	enum index_space { rows, cols, nnz_diag, nnz_offd, rowsp1 };
	using index_spaces = has<rows, cols, nnz_diag, nnz_offd, rowsp1>;

	using connectivities = list<>;

	enum entity_list { owned, shared, ghost };
	using entity_lists = list<
		entity<cols, has<owned>>>;

	template<class T, index_space ispace>
	using def = typename flecsi::field<T>::template definition<par, ispace>;
	static inline const def<std::size_t, rowsp1> rowptr_diag, rowptr_offd;
	static inline const def<std::size_t, nnz_diag> colind_diag;
	static inline const def<std::size_t, nnz_offd> colind_offd;
	static inline const def<double, nnz_diag> values_diag;
	static inline const def<double, nnz_offd> values_offd;
	template<class T, index_space ispace>
	using iacc = typename flecsi::field<T>::template accessor<flecsi::wo, flecsi::wo, flecsi::na>;

	struct init {
		std::vector<csr<double>> local_rows;
		std::size_t ncols_global;
		std::size_t ncolors;
	};

	template<class B> struct interface : B {
		template<index_space Space>
		auto dofs() {
			auto size = B::template special_entities<Space, entity_list::owned>().size();
			return flecsi::topo::make_ids<Space>(
				flecsi::util::iota_view<flecsi::util::id>(0, size));
		}
	};

	static coloring color(const std::string & fname,
	                      init & save) {
		io::matrix_market<>::definition def{fname.c_str()};

		save.ncolors = flecsi::processes();
		detail::coloring_utils cu(def,
		                          {static_cast<flecsi::Color>(save.ncolors),
			                         {3, core::template index<index_space::rows>},
				                         1,
				                         {0, core::template index<index_space::cols>},
				                         {
					                         {1, core::template index<index_space::nnz_diag>},
					                         {2, core::template index<index_space::nnz_offd>},
					                         {4, core::template index<index_space::rowsp1>}}},
		                          MPI_COMM_WORLD);
		cu.dist_graph();
		cu.color_columns();
		cu.color_rows();
		cu.color_nnz();

		save.local_rows = cu.send_rows();
		save.ncols_global = cu.num_cols();

		return cu.generate();
	}


	static void initialize(flecsi::data::topology_slot<par> & s,
	                       const coloring & c, const init & saved) {
		auto lm = flecsi::data::launch::make(s);
		flecsi::execute<init_local_mats, flecsi::mpi>(lm,
		                                              rowptr_diag(lm), colind_diag(lm), values_diag(lm),
		                                              rowptr_offd(lm), colind_offd(lm), values_offd(lm),
		                                              s->reverse_maps_.get<cols>(),
		                                              saved.local_rows, saved.ncols_global, saved.ncolors);

		init_list<index_space::cols, entity_list::owned>(s, c);
	}

	template<entity_list E>
	static void allocate_list(
		flecsi::data::multi<flecsi::topo::array<par>::accessor<flecsi::wo>> aa,
		const std::vector<base::process_coloring> & vpc) {
		auto it = vpc.begin();
		for(auto & a : aa.accessors()) {
			if constexpr(E == owned) {
				a.size() = it++->coloring.owned.size();
			}
			else if(E == shared) {
				a.size() = it++->coloring.shared.size();
			}
			else if(E == ghost) {
				a.size() = it++->coloring.ghost.size();
			}
		}
	}

	template<entity_list E>
	static void populate_list(
		flecsi::data::multi<flecsi::field<flecsi::util::id>::accessor<flecsi::wo>>
		m,
		const std::vector<base::process_coloring> & vpc,
		const base::reverse_maps_t & rmaps) {
		auto it = vpc.begin();
		std::size_t c = 0;
		for(auto & a : m.accessors()) {
			std::size_t i{0};
			if constexpr(E == owned) {
				for(auto e : it++->coloring.owned) {
					a[i++] = rmaps[c].at(e);
				}
			}
			else if(E == shared) {
				for(auto e : it++->coloring.shared) {
					a[i++] = rmaps[c].at(e.id);
				}
			}
			else if(E == ghost) {
				for(auto e : it++->coloring.ghost) {
					a[i++] = rmaps[c].at(e.id);
				}
			}
			c++;
		}
	}

	template<index_space I, entity_list E>
	static void init_list(flecsi::data::topology_slot<par> & s,
	                      const coloring & c) {
		using namespace flecsi;
		auto & el = s->special_.get<I>().template get<E>();
		{
			auto slm = data::launch::make(el);
			execute<allocate_list<E>, flecsi::mpi>(slm, c.idx_spaces[core::index<I>]);
			el.resize();
		}
		auto slm = data::launch::make(el);
		const auto & rmaps = s->reverse_map<I>();
		execute<populate_list<E>>(
			core::special_field(slm), c.idx_spaces[core::index<I>], rmaps);
	}


	static void init_local_mats(
		flecsi::data::multi<
		typename par::accessor<flecsi::ro, flecsi::ro, flecsi::ro>> m,
		flecsi::data::multi<iacc<std::size_t, rowsp1>> mrowptr_diag,
		flecsi::data::multi<iacc<std::size_t, nnz_diag>> mcolind_diag,
		flecsi::data::multi<iacc<double, nnz_diag>> mvalues_diag,
		flecsi::data::multi<iacc<std::size_t, rowsp1>> mrowptr_offd,
		flecsi::data::multi<iacc<std::size_t, nnz_offd>> mcolind_offd,
		flecsi::data::multi<iacc<double, nnz_offd>> mvalues_offd,
		const flecsi::topo::unstructured_base::reverse_maps_t & rmap,
		const std::vector<csr<double>> & local_rows,
		std::size_t ncols_global, std::size_t ncolors) {
		auto rowptr_diaga = mrowptr_diag.accessors();
		auto colind_diaga = mcolind_diag.accessors();
		auto values_diaga = mvalues_diag.accessors();

		auto rowptr_offda = mrowptr_offd.accessors();
		auto colind_offda = mcolind_offd.accessors();
		auto values_offda = mvalues_offd.accessors();

		auto [rank, size] = flecsi::util::mpi::info(MPI_COMM_WORLD);
		const auto pm = flecsi::util::equal_map(ncolors, size);
		const auto cm = flecsi::util::equal_map(ncols_global, ncolors);
		for (unsigned int c = 0; c < m.depth(); ++c) {
			auto gco = pm[rank][c];
			auto & cmat = local_rows[c];
			rowptr_diaga[c][0] = 0;
			rowptr_offda[c][0] = 0;

			for (std::size_t i{0}; i < cmat.rows(); ++i) {
				std::size_t nnz_row_offd = 0;
				std::size_t nnz_row_diag = 0;
				for (std::size_t off = cmat.offsets()[i]; off < cmat.offsets()[i+1]; ++off) {
					auto cid = cmat.indices()[off];
					auto insert = [&](std::size_t base, auto & colind, auto & values, std::size_t & nnz) {
						auto j = base + nnz++;
						colind[c][j] = rmap[c].at(cid);
						values[c][j] = cmat.values()[off];
					};
					if (cm.bin(cid) != gco) { // ghost column (put in offd)
						insert(rowptr_offda[c][i], colind_offda, values_offda, nnz_row_offd);
					} else {
						insert(rowptr_diaga[c][i], colind_diaga, values_diaga, nnz_row_diag);
					}
				}
				rowptr_diaga[c][i+1] = rowptr_diaga[c][i] + nnz_row_diag;
				rowptr_offda[c][i+1] = rowptr_offda[c][i] + nnz_row_offd;
			}
		}
	}
};


}

#endif
