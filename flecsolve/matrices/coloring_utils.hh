#ifndef FLECSOLVE_MATRICES_COLORING_UTILS_HH
#define FLECSOLVE_MATRICES_COLORING_UTILS_HH

#include <flecsi/flog.hh>
#include <flecsi/topo/types.hh>
#include <flecsi/topo/unstructured/types.hh>
#include <flecsi/util/color_map.hh>
#include <flecsi/util/common.hh>
#include <flecsi/util/crs.hh>
#include <flecsi/util/mpi.hh>
#include <flecsi/util/serialize.hh>
#include <flecsi/util/set_utils.hh>

#include "flecsolve/matrices/seq.hh"

namespace flecsolve::mat::detail {


namespace impl = flecsi::topo::unstructured_impl;
namespace util = flecsi::util;

// coloring definition removed from FleCSI
struct coloring_definition {
	struct index_map {
		impl::entity_kind kind;
		impl::entity_index_space idx;
	};

	/// Total number of colors.
	flecsi::Color colors;
	/// Index of primary entity in \c index_spaces.
	/// \warning Not an \c index_space enumerator \b value.
	index_map cid;
	/// Number of layers of ghosts needed.
	std::size_t depth;
	/// Index of vertices in \c index_spaces.
	index_map vid;
	/// Indices of auxiliary entities in \c index_spaces.
	std::vector<index_map> aidxs;
};


template<class MD>
struct coloring_utils {
	using entity_kind = impl::entity_kind;
	using unstructured_base = flecsi::topo::unstructured_base;
	using shared_entity = flecsi::topo::unstructured_impl::shared_entity;
	using ghost_entity = flecsi::topo::unstructured_impl::ghost_entity;

	coloring_utils(MD & md,
	               const coloring_definition & cd,
	               MPI_Comm comm = MPI_COMM_WORLD) :
		md_(md), cd_(cd), comm_(comm) {
		std::tie(rank_, size_) = util::mpi::info(comm_);

		coloring_.comm = comm_;
		coloring_.colors = cd_.colors;

		coloring_.peers.resize(5);
		coloring_.partitions.resize(5);
		coloring_.idx_spaces.resize(5);
	}

	// distribute graph
	void dist_graph();

	void color_columns();

	void color_rows();

	void color_nnz();

	// sends local rows to specialization init
	std::vector<csr<double>> send_rows();

	auto & generate();

	auto num_rows() const {
		return md_.num_rows();
	}

	auto num_cols() const {
		return md_.num_cols();
	}

	auto & connectivity() {
		return connectivity_;
	}

protected:
	auto & row_coloring() {
		return coloring_.idx_spaces[cd_.cid.idx];
	}

	auto & col_coloring() {
		return coloring_.idx_spaces[cd_.vid.idx];
	}

	auto & diag_nnz_coloring() {
		return coloring_.idx_spaces[cd_.aidxs[0].idx];
	}

	auto & offd_nnz_coloring() {
		return coloring_.idx_spaces[cd_.aidxs[1].idx];
	}

	auto & rowp1_coloring() {
		return coloring_.idx_spaces[cd_.aidxs[2].idx];
	}

	auto colors() const {
		return cd_.colors;
	}

	auto colmap() const {
		return util::equal_map(num_cols(), colors());
	}

	auto rowmap() const {
		return util::equal_map(num_rows(), colors());
	}

	auto procmap() const {
		return util::equal_map(colors(), size_);
	}

	auto & local_colors() const {
		return p2co_;
	}

	MD & md_;
	coloring_definition cd_;
	unstructured_base::coloring coloring_;
	std::vector<std::set<flecsi::Color>> color_peers_;
	MPI_Comm comm_;
	std::size_t rank_;
	std::size_t size_;
	std::vector<util::crs> connectivity_;
	std::vector<std::size_t> p2co_;  // process to color map
};


template<class MD>
void coloring_utils<MD>::dist_graph() {
	const auto rm = rowmap();
	const auto pm = procmap();
	connectivity() = util::mpi::one_to_allv(
		[this,pm,rm](int r,int) {
			std::vector<util::crs> graphs;
			for (auto c : pm[r]) {
				graphs.push_back(md_.graph(rm[c]));
			}
			return graphs;
		});

	for (auto c : pm[rank_]) p2co_.push_back(c);

	[this](auto &... c) {
		(c.resize(local_colors().size()),...);
	}(row_coloring(), col_coloring(), diag_nnz_coloring(), offd_nnz_coloring(), rowp1_coloring(), color_peers_);
}


template<class MD>
void coloring_utils<MD>::color_columns() {
	const auto cm = colmap();
	const auto pm = procmap();

	std::vector<std::vector<std::pair<std::size_t, util::gid>>> refs(size_);
	std::vector<std::set<util::gid>> ghost;
	ghost.resize(local_colors().size());
	{
		std::size_t lco{0};
		for (const auto & conn : connectivity()) {
			auto gco = local_colors()[lco];
			for (auto row : conn) {
				for (auto cid :  row) {
					auto owner = cm.bin(cid);
					if (owner != gco) {
						refs[pm.bin(owner)].emplace_back(gco, cid);
						ghost[lco].insert(cid);
					}
				}
			}
			++lco;
		}
	}

	// request referencing colors
	auto rrefs = util::mpi::all_to_allv(
		[&refs](int r, int) -> auto & {
			return refs[r];
		}, comm_);

	// enumerate shared from referencers
	std::map<util::gid, std::set<std::size_t>> shared;
	for (std::size_t r = 0; r < size_; ++r) {
		if (r != rank_) {
			for (auto & ref : rrefs[r]) {
				shared[ref.second].insert(ref.first);
			}
		}
	}

	// populate coloring
	{
		auto & partitions = coloring_.partitions[cd_.vid.idx];
		std::vector<flecsi::Color> process_colors;
		std::vector<std::vector<flecsi::Color>> is_peers(local_colors().size());
		for (std::size_t lco{0}; lco < local_colors().size(); ++lco) {
			process_colors.emplace_back(local_colors()[lco]);
			auto & pc = col_coloring()[lco];
			pc.color = local_colors()[lco];
			pc.entities = num_cols();

			for (auto col : cm[pc.color]) {
				if (shared.count(col)) {
					pc.coloring.shared.emplace_back(shared_entity{
							col, {shared[col].begin(), shared[col].end()}});
				} else {
					pc.coloring.exclusive.emplace_back(col);
				}
				pc.coloring.owned.emplace_back(col);
				pc.coloring.all.emplace_back(col);
			}


			if (ghost[lco].size()) {
				std::set<std::size_t> peers;
				for (auto cid : ghost[lco]) {
					auto owner = cm.bin(cid);
					auto [pr, lco] = pm.invert(owner);
					pc.coloring.ghost.emplace_back(ghost_entity{cid, pr, flecsi::Color(lco), owner});
					pc.coloring.all.emplace_back(cid);
					peers.insert(owner);
				}
				color_peers_[lco].insert(peers.begin(), peers.end());
				is_peers[lco].resize(peers.size());
				std::copy(peers.begin(), peers.end(), is_peers[lco].begin());
				pc.peers.resize(peers.size());
				std::copy(peers.begin(), peers.end(), pc.peers.begin());
			}

			flog_assert(pc.coloring.owned.size() ==
			            pc.coloring.exclusive.size() + pc.coloring.shared.size(),
			            "exclusive and shared primaries != owned primaries");


			// util::force_unique(pc.coloring.all);
			util::force_unique(pc.coloring.owned);
			util::force_unique(pc.coloring.exclusive);
			util::force_unique(pc.coloring.shared);
			util::force_unique(pc.coloring.ghost);

			partitions.emplace_back(pc.coloring.all.size());
		}

		// gather tight peer information for columns
		{
			auto & peers = coloring_.peers[cd_.vid.idx];
			peers.reserve(cd_.colors);

			for (auto vp : util::mpi::all_gatherv(is_peers, comm_)) {
				for (auto & pc : vp) { // over process colors
					peers.push_back(std::move(pc));
				}
			}
		}

		// gather process to color mapping (TODO: can compute to avoid communication)
		coloring_.process_colors = util::mpi::all_gatherv(process_colors, comm_);

		// gather partition sizes
		flecsi::topo::concatenate(partitions, cd_.colors, comm_);
	}
}


template<class MD>
void coloring_utils<MD>::color_rows() {
	const auto rm = rowmap();

	{ //row coloring
		// populate coloring
		auto & partitions = coloring_.partitions[cd_.cid.idx];
		for (std::size_t lco{0}; lco < local_colors().size(); ++lco) {
			auto & pc = row_coloring()[lco];
			pc.color = local_colors()[lco];
			pc.entities = num_rows();
			[&](auto & ... v) {
				(v.resize(rm[pc.color].size()), ...);
				// just fill with unique values
				// (std::iota(v.begin(), v.end(), rm[pc.color][0] + pc.color), ...);
				(std::iota(v.begin(), v.end(), 0), ...);
			}(pc.coloring.owned,
			  pc.coloring.exclusive,
			  pc.coloring.all);

			partitions.emplace_back(pc.coloring.all.size());
		}

		auto & peers = coloring_.peers[cd_.cid.idx];
		peers.reserve(cd_.colors);
		for (std::size_t i{0}; i < cd_.colors; ++i)
			peers.emplace_back();

		// gather partition sizes
		flecsi::topo::concatenate(partitions, cd_.colors, comm_);
	}

	{ // rowp1 coloring
		// populate coloring
		auto & partitions = coloring_.partitions[cd_.aidxs[2].idx];
		for (std::size_t lco{0}; lco < local_colors().size(); ++lco) {

			auto & pc = rowp1_coloring()[lco];
			pc.color = local_colors()[lco];
			pc.entities = num_rows() + coloring_.colors;
			[&](auto & ... v) {
				(v.resize(rm[pc.color].size() + 1), ...);
				// just fill with unique values
				// (std::iota(v.begin(), v.end(), rm[pc.color][0] + pc.color), ...);
				(std::iota(v.begin(), v.end(), 0), ...);
			}(pc.coloring.owned,
			  pc.coloring.exclusive,
			  pc.coloring.all);

			partitions.emplace_back(pc.coloring.all.size());
		}

		auto & peers = coloring_.peers[cd_.aidxs[2].idx];
		peers.reserve(cd_.colors);
		for (std::size_t i{0}; i < cd_.colors; ++i)
			peers.emplace_back();

		// gather partition sizes
		flecsi::topo::concatenate(partitions, cd_.colors, comm_);
	}
}


template<class MD>
void coloring_utils<MD>::color_nnz() {
	const auto cm = colmap();

	std::vector<int> diag_size(local_colors().size());
	std::vector<int> offd_size(local_colors().size());
	int proc_diag{0};
	int proc_offd{0};
	for (std::size_t lco{0}; lco < local_colors().size(); ++lco) {
		const auto & conn = connectivity()[lco];
		auto gco = local_colors()[lco];
		for (auto row : conn) {
			for (auto cid : row) {
				auto owner = cm.bin(cid);
				if (owner != gco) {
					++offd_size[lco];
					++proc_offd;
				} else {
					++diag_size[lco];
					++proc_diag;
				}
			}
		}
	}

	// this is really dumb (should use a subtopology instead of coloring nnz)
	int global_offd, global_diag;
	MPI_Allreduce(&proc_diag, &global_diag, 1, MPI_INT, MPI_SUM, comm_);
	MPI_Allreduce(&proc_offd, &global_offd, 1, MPI_INT, MPI_SUM, comm_);

	// populate coloring
	auto pop = [this](
		auto & nnz_col,
		auto & partitions,
		auto & peers,
		const auto & lsize,
		int gsize) {
		(void)lsize;
		for (std::size_t lco{0}; lco < local_colors().size(); ++lco) {
			auto & pc = nnz_col[lco];
			pc.color = local_colors()[lco];
			pc.entities = gsize;
			[&](auto & ... v) {
				(v.resize(lsize[lco]), ...);
				// // just fill with unique values
				(std::iota(v.begin(), v.end(), 0), ...);
			}(pc.coloring.owned,
			  pc.coloring.exclusive,
			  pc.coloring.all);
			partitions.emplace_back(pc.coloring.all.size());
		}

		peers.reserve(cd_.colors);
		for (std::size_t i{0}; i < cd_.colors; ++i)
			peers.emplace_back();

		// gather partition sizes
		flecsi::topo::concatenate(partitions, cd_.colors, comm_);
	};
	pop(diag_nnz_coloring(),
	    coloring_.partitions[cd_.aidxs[0].idx],
	    coloring_.peers[cd_.aidxs[0].idx],
	    diag_size, global_diag);
	pop(offd_nnz_coloring(),
	    coloring_.partitions[cd_.aidxs[1].idx],
	    coloring_.peers[cd_.aidxs[1].idx],
	    offd_size, global_offd);
}


template<class MD>
auto & coloring_utils<MD>::generate() {
	// gather peer information
	auto & cp = coloring_.color_peers;
	cp.reserve(color_peers_.size());
	for (const auto & s : color_peers_)
		cp.push_back(s.size());
	flecsi::topo::concatenate(cp, cd_.colors, comm_);

	return coloring_;
}


template<class MD>
std::vector<csr<double>> coloring_utils<MD>::send_rows() {
	const auto rm = rowmap();
	const auto pm = procmap();
	// assuming we have not repartitioned
	return util::mpi::one_to_allv(
		[=](int r, int) {
			std::vector<csr<double>> lmats;
			for (auto c : pm[r]) {
				lmats.push_back(md_.matrix(rm[c]));
			}
			return lmats;
		});
}

}
// Serialization rules removed from FleCSI.
namespace flecsi {
template<>
struct util::serial::traits<topo::unstructured_impl::shared_entity> {
  using type = topo::unstructured_impl::shared_entity;
  template<class P>
  static void put(P & p, const type & s) {
    serial::put(p, s.id, s.dependents);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{r, r};
  }
};

template<>
struct util::serial::traits<topo::unstructured_impl::index_coloring> {
  using type = topo::unstructured_impl::index_coloring;
  template<class P>
  static void put(P & p, const type & c) {
    serial::put(p, c.all, c.owned, c.exclusive, c.shared, c.ghost);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{r, r, r, r, r};
  }
};

template<>
struct util::serial::traits<topo::unstructured_impl::process_coloring> {
  using type = topo::unstructured_impl::process_coloring;
  template<class P>
  static void put(P & p, const type & c) {
    serial::put(p,
      c.color,
      c.entities,
      c.coloring,
      c.peers,
      c.cnx_allocs,
      c.cnx_colorings);
  }
  static type get(const std::byte *& p) {
    const cast r{p};
    return type{r, r, r, r, r, r};
  }
};
}
#endif
