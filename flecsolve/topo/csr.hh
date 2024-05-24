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
#ifndef FLECSOLVE_TOPO_CSR_H
#define FLECSOLVE_TOPO_CSR_H

#include <unordered_map>
#include <variant>

#include "flecsi/topo/types.hh"
#include "flecsi/topo/core.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/crs.hh"
#include "flecsi/data/copy_plan.hh"
#include "flecsi/data/map.hh"
#include "flecsi/data/layout.hh"

#include "flecsolve/matrices/seq.hh"

namespace flecsolve::topo {

namespace csr_impl {
struct process_coloring {
	flecsi::Color color;
	flecsi::util::id entities;
};

struct metadata {
	MPI_Comm comm;
	flecsi::util::gid nrows, ncols;
	struct rng {
		flecsi::util::gid beg, end;
		constexpr flecsi::util::gid size() const { return end - beg + 1; }
	};
	rng rows;
	rng cols;
};

struct partition {
	partition() : storage{flecsi::util::equal_map(1, 1)} {}

	constexpr auto operator[](flecsi::Color c) const {
		return std::visit([=](auto & part) { return part[c]; }, storage);
	}

	constexpr flecsi::Color bin(std::size_t i) const {
		return std::visit([=](auto & part) { return part.bin(i); }, storage);
	}

	constexpr auto invert(std::size_t i) const {
		return std::visit([=](auto & part) { return part.invert(i); }, storage);
	}

	void set_block_map(std::size_t s, flecsi::Color bins) {
		storage = flecsi::util::equal_map(s, bins);
	}

	void set_offsets(const flecsi::util::offsets & off) { storage = off; }

	constexpr flecsi::Color size() const {
		return std::visit([=](auto & part) { return part.size(); }, storage);
	}

protected:
	std::variant<flecsi::util::equal_map, flecsi::util::offsets> storage;
};
}
// everything that does not depend on policy
struct csr_base {
	using process_coloring = csr_impl::process_coloring;
	using partition = csr_impl::partition;
	using metadata = csr_impl::metadata;
	using destination_intervals = std::vector< // local colors
		std::vector< // contiguous intervals
			flecsi::data::subrow>>;
	using source_pointers = std::vector< // local colors
		std::map< // global source colors
			flecsi::Color,
			std::vector< // color source pointers
				std::pair<flecsi::util::id /* local ghost offset */,
	                      flecsi::util::id /* remote shared offset */>>>>;
	struct coloring {
		MPI_Comm comm;
		flecsi::Color colors;
		flecsi::util::gid nrows;
		flecsi::util::gid ncols;
		std::vector< // index spaces
			std::vector< // process colors
				process_coloring>>
			idx_spaces;
		std::vector< // process colors
			std::vector<flecsi::util::gid>>
			column_ghosts;
		partition row_part;
		partition col_part;
	};

	static std::size_t idx_size(std::vector<std::size_t> vs, std::size_t c) {
		return vs[c];
	}

	static void
	idx_itvls(flecsi::Color colors,
	          const csr_impl::partition & cm,
	          const std::vector<std::vector<flecsi::util::gid>> ghosts,
	          destination_intervals & intervals,
	          source_pointers & pointers,
	          MPI_Comm comm) {
		auto [rank, comm_size] = flecsi::util::mpi::info(comm);
		const flecsi::util::equal_map pm(colors, comm_size);
		pointers.resize(ghosts.size());
		intervals.resize(ghosts.size());
		{
			auto it = ghosts.begin();
			auto pit = pointers.begin();
			auto iit = intervals.begin();
			for (auto col : pm[rank]) {
				const auto & cghosts = *it++;
				auto & pts = *pit++;
				std::size_t off{0};
				for (auto cid : cghosts) {
					auto [owner, shared_offset] = cm.invert(cid);
					pts[owner].emplace_back(
						std::make_pair(cm[col].size() + off++, shared_offset));
				}

				auto & inter = *iit++;
				inter.emplace_back(std::make_pair(
					cm[col].size(), cm[col].size() + cghosts.size()));
			}
		}
	}

	static void
	set_dests(flecsi::data::multi<flecsi::field<
				  flecsi::data::intervals::Value>::accessor<flecsi::wo>> aa,
	          const std::vector<std::vector<flecsi::data::subrow>> & intervals,
	          const MPI_Comm &) {
		std::size_t ci = 0;
		for (auto [c, a] : aa.components()) {
			auto & iv = intervals[ci++];
			flog_assert(a.span().size() == iv.size(),
			            "interval size mismatch a.span (" << a.span().size()
			                                              << ") != intervals ("
			                                              << iv.size() << ")");
			std::size_t i{0};
			for (auto & it : iv) {
				a[i++] = flecsi::data::intervals::make(it, c);
			}
		}
	}

	template<flecsi::PrivilegeCount N>
	static void
	set_ptrs(flecsi::data::multi<
	         flecsi::field<flecsi::data::copy_engine::Point>::accessor1<
					 flecsi::privilege_repeat<flecsi::wo, N>>> aa,
	         const std::vector<std::map<
				 flecsi::Color,
				 std::vector<std::pair<flecsi::util::id, flecsi::util::id>>>> &
	             points,
	         const MPI_Comm &) {
		std::size_t ci = 0;
		for (auto & a : aa.accessors()) {
			for (auto const & [owner, ghosts] : points[ci++]) {
				for (auto const & [local_offset, remote_offset] : ghosts) {
					// si.first: owner
					// p.first: local ghost offset
					// p.second: remote shared offset
					a[local_offset] = flecsi::data::copy_engine::point(owner, remote_offset);
				}
			}
		}
	}
};

template<class P>
struct csr_category : csr_base, flecsi::topo::with_meta<P> {
	using index_space = typename P::index_space;
	using index_spaces = typename P::index_spaces;

	flecsi::Color colors() const { return part_.front().colors(); }

	template<index_space S>
	static constexpr std::size_t index = index_spaces::template index<S>;

	template<flecsi::Privileges>
	struct access;

	template<index_space S>
	flecsi::data::region & get_region() {
		return part_.template get<S>();
	}

	template<index_space S>
	flecsi::topo::repartition & get_partition() {
		return part_.template get<S>();
	}

	template<typename Type,
	         flecsi::data::layout Layout,
	         typename P::index_space Space>
	[[nodiscard]] const flecsi::data::copy_plan * ghost_copy(
		const flecsi::data::field_reference<Type, Layout, P, Space> &) {
		static_assert(Space == P::column_space);
		return &column_plan_;
	}

	csr_category(const coloring & c)
		: csr_category(
			  [&c]() -> auto & {
				  flog_assert(
					  c.idx_spaces.size() == index_spaces::size,
					  "invalid number of idx_spaces: " << c.idx_spaces.size());
				  return c;
			  }(),
			  index_spaces()) {}

private:
	template<auto... VV>
	csr_category(const csr_base::coloring & c, flecsi::util::constants<VV...>)
		: flecsi::topo::with_meta<P>(c.colors),
		  part_{{flecsi::topo::make_repartitioned<P, VV>(
			  c.colors,
			  flecsi::make_partial<idx_size>([&]() {
				  std::vector<std::size_t> partitions;
				  for (const auto & pc : c.idx_spaces[index<VV>]) {
					  partitions.push_back(pc.entities);
				  }
				  flecsi::topo::concatenate(partitions, c.colors, c.comm);
				  return partitions;
			  }()))...}},
		  column_plan_{make_copy_plan(c, part_[index<P::column_space>])} {
		auto lm = flecsi::data::launch::make(this->meta);
		flecsi::execute<set_meta, flecsi::mpi>(metadata_field(lm), c);
	}

	flecsi::data::copy_plan make_copy_plan(const csr_base::coloring & c,
	                                       flecsi::topo::repartitioned &) {
		std::vector<std::size_t> num_intervals(c.colors, 1);
		destination_intervals intervals;
		source_pointers pointers;

		flecsi::execute<idx_itvls, flecsi::mpi>(
			c.colors, c.col_part, c.column_ghosts, intervals, pointers, c.comm);

		auto dest_task = [&](auto f) {
			auto lm = flecsi::data::launch::make(f.topology());
			flecsi::execute<set_dests, flecsi::mpi>(lm(f), intervals, c.comm);
		};

		auto ptrs_task = [&](auto f) {
			auto lm = flecsi::data::launch::make(f.topology());
			flecsi::execute<
				set_ptrs<P::template privilege_count<P::column_space>>,
				flecsi::mpi>(lm(f), pointers, c.comm);
		};

		return {*this,
		        num_intervals,
		        dest_task,
		        ptrs_task,
		        flecsi::util::constant<P::column_space>()};
	}

	static void set_meta(
		flecsi::data::multi<
			flecsi::field<metadata, flecsi::data::single>::accessor<flecsi::wo>>
			mm,
		const csr_base::coloring & c) {
		const auto ma = mm.accessors();
		auto [rank, comm_size] = flecsi::util::mpi::info(c.comm);

		const auto & cm = c.col_part;
		const auto & rm = c.row_part;
		const flecsi::util::equal_map pm(c.colors, comm_size);
		assert(static_cast<std::size_t>(ma.size()) == pm[rank].size());
		std::size_t i{0};
		for (flecsi::Color col : pm[rank]) {
			auto & e = ma[i++].get();
			e.comm = c.comm;
			e.nrows = c.nrows;
			e.ncols = c.ncols;
			e.cols.beg = cm[col].front();
			e.cols.end = cm[col].back();
			e.rows.beg = rm[col].front();
			e.rows.end = rm[col].back();
		}
	}

	flecsi::util::key_array<flecsi::topo::repartitioned, index_spaces> part_;
	flecsi::data::copy_plan column_plan_;

	static inline const flecsi::field<metadata, flecsi::data::single>::
		definition<flecsi::topo::meta<P>>
			metadata_field;

	template<class T, index_space ispace>
	using field_def = typename flecsi::field<T>::template definition<P, ispace>;

	template<index_space nnz_space>
	struct csr_def {
		static inline const field_def<typename P::size_type,
		                              index_space::rowsp1>
			offsets;
		static inline const field_def<typename P::size_type, nnz_space> indices;
		static inline const field_def<typename P::scalar_type, nnz_space>
			values;
	};

	using diag = csr_def<index_space::nnz_diag>;
	using offd = csr_def<index_space::nnz_offd>;

	static inline const field_def<flecsi::util::gid, index_space::nnz_offd> colmap_field;
};

template<class P>
template<flecsi::Privileges Priv>
struct csr_category<P>::access {
	template<class F>
	void send(F && f) {
		send_csr<csr_category::diag>(f, diag_);
		send_csr<csr_category::offd>(f, offd_);
		const auto meta = [](auto & n) -> auto & { return n.meta; };
		meta_.topology_send(f, meta);
		f(colmap_, [](auto & t) { return csr_category::colmap_field(t.get()); });
	}

	template<class FS, class A, class F>
	void send_csr(F && f, A & a) {
		f(a.offsets, [](auto & t) { return FS::offsets(t.get()); });
		f(a.indices, [](auto & t) { return FS::indices(t.get()); });
		f(a.values, [](auto & t) { return FS::values(t.get()); });
	}

	FLECSI_INLINE_TARGET auto diag() { return diag_; }

	FLECSI_INLINE_TARGET auto offd() { return offd_; }

	FLECSI_INLINE_TARGET auto colmap() { return colmap_; }

	FLECSI_INLINE_TARGET const csr_impl::metadata & meta() { return *meta_; }

private:
	flecsi::data::scalar_access<csr_category<P>::metadata_field, Priv> meta_;

	template<const auto & Field>
	using accessor = flecsi::data::accessor_member<
		Field,
		flecsi::privilege_pack<flecsi::privilege_merge(Priv)>>;
	template<class C>
	struct csr_acc {
		accessor<C::offsets> offsets;
		accessor<C::indices> indices;
		accessor<C::values> values;
	};

	csr_acc<csr_category::diag> diag_;
	csr_acc<csr_category::offd> offd_;
	accessor<csr_category::colmap_field> colmap_;
};
}
namespace flecsi::topo {
template<>
struct detail::base<flecsolve::topo::csr_category> {
	using type = flecsolve::topo::csr_base;
};
}

namespace flecsolve::topo {
template<class scalar, class size = std::size_t>
struct csr : flecsi::topo::help,
			 flecsi::topo::specialization<csr_category, csr<scalar, size>> {
	using coloring = csr_base::coloring;
	using size_type = size;
	using scalar_type = scalar;
	using csr_t = mat::csr<scalar, size>;
	enum index_space { rows, cols, nnz_diag, nnz_offd, rowsp1 };
	using index_spaces = has<rows, cols, nnz_diag, nnz_offd, rowsp1>;
	static constexpr index_space column_space = cols;

	template<index_space S>
	using vec_def =
		typename flecsi::field<scalar>::template definition<csr<scalar, size>,
	                                                        S>;

	template<index_space S>
	static constexpr flecsi::PrivilegeCount
		privilege_count = (S == index_space::cols) ? 2 : 1;

	template<class B>
	struct interface : B {
		FLECSI_INLINE_TARGET auto diag() {
			auto diaga = B::diag();
			return mat::csr_view(diaga.offsets.span(),
			                     diaga.indices.span(),
			                     diaga.values.span());
		}

		FLECSI_INLINE_TARGET auto offd() {
			auto offda = B::offd();
			return mat::csr_view(offda.offsets.span(),
			                     offda.indices.span(),
			                     offda.values.span());
		}

		FLECSI_INLINE_TARGET const auto & meta() { return B::meta(); }
		FLECSI_INLINE_TARGET auto colmap() { return B::colmap(); }

		template<index_space S>
		auto dofs() {
			static_assert(S == index_space::rows || S == index_space::cols);
			auto view = (S == index_space::rows) ? meta().rows : meta().cols;
			return flecsi::topo::make_ids<S>(
				flecsi::util::iota_view<flecsi::util::id>(0, view.size()));
		}

		template<index_space S>
		flecsi::util::gid global_id(flecsi::topo::id<S> lid) {
			static_assert(S == index_space::rows || S == index_space::cols);
			auto view = (S == index_space::rows) ? meta().rows : meta().cols;
			auto lidi = static_cast<flecsi::util::id>(lid);
			if constexpr (S == index_space::cols) {
				if (lidi >= view.size()) {
					return colmap()[lidi - view.size()];
				}
			}
			return view.beg + static_cast<flecsi::util::id>(lid);
		}
	};

	struct init {
		MPI_Comm comm;
		std::size_t nrows, ncols;
		csr_impl::partition row_part, col_part;
		std::vector<csr_t> proc_mats;
	};

	static coloring color(const init & ci) {
		coloring c;

		c.comm = ci.comm;
		c.colors = ci.row_part.size();
		c.nrows = ci.nrows;
		c.ncols = ci.ncols;
		c.row_part = ci.row_part;
		c.col_part = ci.col_part;

		auto [rank, comm_size] = flecsi::util::mpi::info(ci.comm);
		const auto & cm = ci.col_part;
		const auto & rm = ci.row_part;
		const flecsi::util::equal_map pm(c.colors, comm_size);

		std::vector<std::array<flecsi::util::id, index_spaces::size>> isizes;
		isizes.resize(pm[rank].size());
		c.column_ghosts.resize(pm[rank].size());
		{ // compute sizes of each index space
			auto lmat = ci.proc_mats.begin();
			auto lsizes = isizes.begin();
			auto lghosts = c.column_ghosts.begin();
			for (const flecsi::Color col : pm[rank]) {
				std::size_t num_diag{0}, num_offd{0};
				auto & mat = *lmat++;
				auto & sz = *lsizes++;
				auto & ghosts = *lghosts++;
				for (std::size_t row = 0; row < mat.data.offsets().size() - 1;
				     ++row) {
					for (std::size_t off = mat.data.offsets()[row];
					     off < mat.data.offsets()[row + 1];
					     ++off) {
						auto cid = mat.data.indices()[off];
						if (cm.bin(cid) != col) {
							++num_offd;
							ghosts.push_back(cid);
						}
						else
							++num_diag;
					}
				}
				flecsi::util::force_unique(ghosts);
				sz[rows] = rm[col].size();
				sz[cols] = cm[col].size() + ghosts.size();
				sz[nnz_diag] = num_diag;
				sz[nnz_offd] = num_offd;
				sz[rowsp1] = sz[rows] + 1;
			}
		}

		// populate index space coloring
		for (std::size_t ispace = 0; ispace < index_spaces::size; ++ispace) {
			c.idx_spaces.emplace_back();
			std::size_t lc = 0;
			for (const flecsi::Color col : pm[rank]) {
				c.idx_spaces.back().push_back({col, isizes[lc++][ispace]});
			}
		}

		return c;
	}

	/*
	  Initialize local diag and offd matrices using coloring input.

	  This initializes diag and offd using the input matrices on each
	  process.  The input matrices are split into diag and offd and
	  the columns in offd are compressed using the ghost ordering set
	  by the coloring.
	*/
	static void
	init_mats(flecsi::data::multi<
				  typename csr<scalar, size>::template accessor<flecsi::wo>> m,
	          const coloring & c,
	          const init & ci) {
		auto [rank, comm_size] = flecsi::util::mpi::info(c.comm);
		const auto & cm = ci.col_part;
		const flecsi::util::equal_map pm(c.colors, comm_size);

		auto ma = m.accessors();
		auto lmat = ci.proc_mats.begin();
		for (std::size_t lc = 0; lc < m.depth(); ++lc) {
			auto col = pm[rank][lc];

			auto & mat = *lmat++;
			auto diag = ma[lc].diag();
			auto offd = ma[lc].offd();
			auto colmap = ma[lc].colmap();

			// initialize reverse map for ghosts
			std::unordered_map<flecsi::util::gid, size> rcolmap;
			for (size i = 0; i < c.column_ghosts[lc].size(); ++i) {
				rcolmap[c.column_ghosts[lc][i]] = i + cm[col].size();
				colmap[i] = c.column_ghosts[lc][i];
			}

			diag.data.offsets()[0] = 0;
			offd.data.offsets()[0] = 0;
			for (size row = 0; row < mat.data.offsets().size() - 1; ++row) {
				size nnz_row_offd{0}, nnz_row_diag{0};
				for (size off = mat.data.offsets()[row];
				     off < mat.data.offsets()[row + 1];
				     ++off) {
					auto cid = mat.data.indices()[off];
					auto val = mat.data.values()[off];
					auto insert = [&](size local_col,
					                  size base,
					                  auto colind,
					                  auto values,
					                  size & nnz) {
						auto j = base + nnz++;
						colind[j] = local_col;
						values[j] = val;
					};
					if (cm.bin(cid) == col) { // insert in diag
						insert(cm.invert(cid).second,
						       diag.data.offsets()[row],
						       diag.data.indices(),
						       diag.data.values(),
						       nnz_row_diag);
					}
					else { // insert in offd
						insert(rcolmap.at(cid),
						       offd.data.offsets()[row],
						       offd.data.indices(),
						       offd.data.values(),
						       nnz_row_offd);
					}
				}
				diag.data.offsets()[row + 1] =
					diag.data.offsets()[row] + nnz_row_diag;
				offd.data.offsets()[row + 1] =
					offd.data.offsets()[row] + nnz_row_offd;
			}
		}
	}

	/*
	  Initialize csr data after topology has been setup.

	  - execute mpi task to set diag and offd matrices
	*/
	static void initialize(flecsi::data::topology_slot<csr> & s,
	                       const coloring & c,
	                       const init & ci) {
		auto lm = flecsi::data::launch::make(s);
		flecsi::execute<init_mats, flecsi::mpi>(lm, c, ci);
	}
};

}

#endif
