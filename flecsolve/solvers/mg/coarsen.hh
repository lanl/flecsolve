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
#ifndef FLECSOLVE_SOLVERS_MG_COARSEN_H
#define FLECSOLVE_SOLVERS_MG_COARSEN_H

#include <limits>

#include <flecsi/util/mpi.hh>

#include "flecsolve/matrices/parcsr.hh"

namespace flecsolve::mg::ua {

namespace task {

using aggregate_t = std::vector<std::vector<std::size_t>>;
using soc_t = std::vector<std::set<std::size_t>>;
using flecsi::data::multi;

template<class scalar, class size>
using csr_acc = multi<typename topo::csr<scalar, size>::template accessor<flecsi::ro>>;
template<flecsi::partition_privilege_t... privs>
using aggt_acc = multi<
	typename flecsi::field<flecsi::util::id>::template accessor<privs...>>;
template<class scalar, class size>
using csr_init = typename topo::csr<scalar, size>::init;

class prospect
{
public:
	std::size_t pop() {
		assert(!needs_rebuild);

		auto selected = prio_node.cbegin();
		node_prio.erase(selected->second);
		prio_node.erase(selected);
		return selected->second;
	}

	void remove(std::size_t e) {
		assert(!needs_rebuild);

		// remove from forward map
		auto eprio = node_prio.at(e);
		node_prio.erase(e);

		// remove from reverse map
		auto rng = prio_node.equal_range(eprio);
		for (auto i = rng.first; i != rng.second; ++i) {
			if (i->second == e) {
				prio_node.erase(i);
				break;
			}
		}
	}

	void decrement_priority(std::size_t e) {
		auto npnode = node_prio.find(e);
		if (npnode != node_prio.end()) {
			auto prio = npnode->second;

			auto rng = prio_node.equal_range(prio);
			for (auto i = rng.first; i != rng.second; ++i) {
				if (i->second == e) {
					prio_node.erase(i);
					break;
				}
			}

			prio_node.insert({prio - 1, e});
			--node_prio[e];
		}
	}

	void insert(std::size_t node) {
		node_prio[node];
		needs_rebuild = true;
	}

	bool contains(std::size_t node) const {
		return node_prio.find(node) != node_prio.end();
	}

	bool empty() const { return node_prio.empty(); }

	void prioritize(const soc_t & soc) {
		for (const auto & strength : soc) {
			for (auto node : strength) {
				++node_prio[node];
			}
		}
		rebuild();
	}

	void dump() {
		for (const auto & v : node_prio) {
			std::cout << "node -> priority: " << v.first << " -> " << v.second
					  << std::endl;
		}

		for (const auto & v : prio_node) {
			std::cout << "priority => node: " << v.first << " => " << v.second
					  << std::endl;
		}
	}

	std::size_t size() const { return node_prio.size(); }

protected:
	void rebuild() {
		for (const auto & nd : node_prio) {
			prio_node.insert({nd.second, nd.first});
		}
		needs_rebuild = false;
	}
	std::multimap<std::size_t, std::size_t> prio_node;
	std::map<std::size_t, std::size_t> node_prio;
	bool needs_rebuild;
};

template<class scalar, class size, template<class> class data>
soc_t soc_graph(float beta,
                const mat::csr<scalar, size, data> & diag,
                const mat::csr<scalar, size, data> & offd,
                bool checkdd,
                const prospect & unmarked) {
	soc_t soc(diag.rows());

	for (std::size_t r = 0; r < diag.rows(); ++r) {
		auto find_max_couple = [=](const auto & mat) {
			auto [rowptr, colind, values] = mat.rep();
			auto min_el = std::min_element(values.begin() + rowptr[r],
			                               values.begin() + rowptr[r + 1]);
			if (min_el == values.end())
				return std::numeric_limits<scalar>::max();
			else
				return *min_el;
		};
		auto max_couple =
			std::min(find_max_couple(diag), find_max_couple(offd));

		auto update_soc = [=, &unmarked](auto & soc, const auto & mat) {
			auto [rowptr, colind, values] = mat.rep();
			for (std::size_t off = rowptr[r]; off < rowptr[r + 1]; ++off) {
				auto c = colind[off];
				if (r != c && (!checkdd || unmarked.contains(c)) &&
				    c < mat.rows()) {
					if (values[off] < (beta * max_couple)) {
						soc[r].insert(c);
					}
				}
			}
		};
		update_soc(soc, diag);
		update_soc(soc, offd);
	}

	return soc;
}

template<class scalar, class size, template<class> class data>
auto get_unmarked(const mat::csr<scalar, size, data> & diag,
                  const mat::csr<scalar, size, data> & offd,
                  bool checkdd) {
	prospect unmarked;

	if (checkdd) {
		for (size r = 0; r < diag.rows(); ++r) {
			auto offd_rowptr = offd.data.offsets();
			auto offd_sum = std::reduce(
				offd.data.values().begin() + offd_rowptr[r],
				offd.data.values().begin() + offd_rowptr[r + 1],
				0,
				[](scalar a, scalar b) { return std::abs(a) + std::abs(b); });
			auto [rowptr, colind, values] = diag.rep();
			scalar diag_val = 0.;
			for (size off = rowptr[r]; off < rowptr[r + 1]; ++off) {
				if (r != colind[off])
					offd_sum += std::abs(values[off]);
				else
					diag_val = values[off];
			}
			if (!(diag_val > 5 * offd_sum))
				unmarked.insert(r);
		}
	}
	else {
		for (size r = 0; r < diag.rows(); ++r) {
			unmarked.insert(r);
		}
	}

	return unmarked;
}

template<class scalar, class size, template<class> class data>
auto find_pair(std::size_t r,
               const prospect & unmarked,
               const mat::csr<scalar, size, data> & diag) {
	auto [rowptr, colind, values] = diag.rep();
	std::pair<scalar, std::size_t> curr;
	curr.first = std::numeric_limits<scalar>::max();
	curr.second = -1;
	for (std::size_t off = rowptr[r]; off < rowptr[r + 1]; ++off) {
		auto c = colind[off];
		auto v = values[off];
		if (unmarked.contains(c) && (v < curr.first)) {
			curr.first = v;
			curr.second = c;
		}
	}
	assert(curr.second >= 0);

	return curr.second;
}

template<class scalar, class size, template<class> class data>
aggregate_t pairwise_agg(float beta,
                         const mat::csr<scalar, size, data> & diag,
                         const mat::csr<scalar, size, data> & offd,
                         bool checkdd) {
	aggregate_t aggregates;

	auto unmarked = get_unmarked(diag, offd, checkdd);
	auto soc = soc_graph(beta, diag, offd, checkdd, unmarked);
	unmarked.prioritize(soc);

	while (!unmarked.empty()) {
		auto selected = unmarked.pop();
		auto pair = find_pair(selected, unmarked, diag);

		auto update_priorities = [&](std::size_t k) {
			for (auto l : soc[k]) {
				unmarked.decrement_priority(l);
			}
		};
		if (soc[selected].find(pair) != soc[selected].end()) {
			unmarked.remove(pair);
			update_priorities(pair);
			aggregates.push_back({selected, pair});
		}
		else
			aggregates.push_back({selected});
		update_priorities(selected);
	}

	return aggregates;
}

template<class T>
void transpose_aggregates(const aggregate_t & agg, T & aggt, std::size_t offset) {
	for (std::size_t i = 0; i < agg.size(); ++i) {
		for (auto fi : agg[i]) {
			aggt[fi] = offset + i;
		}
	}
}

inline auto local_aggregate_transpose(const aggregate_t & agg) {
	std::map<std::size_t, std::size_t> aggt;

	transpose_aggregates(agg, aggt, 0);

	return aggt;
}

template<class scalar, class size, template<class> class data>
auto create_aux(const mat::csr<scalar, size, data> & diag,
                const mat::csr<scalar, size, data> & offd,
                const aggregate_t & agg) {
	mat::csr<scalar, size> diagc(agg.size(), agg.size());
	mat::csr<scalar, size> offdc(agg.size(), offd.cols());

	auto aggt = local_aggregate_transpose(agg);
	{ // collapse diag
		auto [rowptr_coarse, colind_coarse, values_coarse] = diagc.data.vecs();
		auto [rowptr, colind, values] = diag.rep();
		for (size rc = 0; rc < agg.size(); ++rc) {
			// coarse column index -> aggregated value
			std::map<size, scalar> agg_values;
			for (auto r : agg[rc]) {
				for (size off = rowptr[r]; off < rowptr[r + 1]; ++off) {
					if (aggt[colind[off]] !=
					    std::numeric_limits<flecsi::util::id>::max())
						agg_values[aggt[colind[off]]] += values[off];
				}
			}

			size off{0};
			for (const auto & ce : agg_values) {
				colind_coarse.push_back(ce.first);
				values_coarse.push_back(ce.second);
				++off;
			}
			rowptr_coarse[rc + 1] = rowptr_coarse[rc] + off;
		}
	}

	{ // collapse offd
		auto [rowptr_coarse, colind_coarse, values_coarse] = offdc.data.vecs();
		auto [rowptr, colind, values] = diag.rep();
		for (size rc = 0; rc < agg.size(); ++rc) {
			// column index -> aggregated value
			std::map<size, scalar> agg_values;
			for (auto r : agg[rc]) {
				for (size off = rowptr[r]; off < rowptr[r + 1]; ++off) {
					agg_values[colind[off]] += values[off];
				}
			}

			size off{0};
			for (const auto & ce : agg_values) {
				colind_coarse.push_back(ce.first);
				values_coarse.push_back(ce.second);
				++off;
			}
			rowptr_coarse[rc + 1] = rowptr_coarse[rc] + off;
		}
	}

	return std::make_pair(std::move(diagc), std::move(offdc));
}

inline aggregate_t agg_union(const aggregate_t & agg1,
                             const aggregate_t & agg2) {
	aggregate_t agg(agg2.size());

	for (std::size_t i = 0; i < agg2.size(); ++i) {
		for (auto e2 : agg2[i]) {
			for (auto e1 : agg1[e2])
				agg[i].push_back(e1);
		}
	}

	return agg;
}

template<class scalar, class size, template<class> class data>
auto double_pairwise_agg(float beta,
                         const mat::csr<scalar, size, data> & diag,
                         const mat::csr<scalar, size, data> & offd,
                         bool checkdd) {
	auto agg1 = pairwise_agg(beta, diag, offd, checkdd);
	auto aux = create_aux(diag, offd, agg1);
	auto agg2 = pairwise_agg(beta, aux.first, aux.second, false);

	return agg_union(agg1, agg2);
}

inline auto create_partition(MPI_Comm comm, const std::vector<aggregate_t> & agg) {
	std::vector<std::size_t> local_agg_sizes;
	for (const auto & a : agg) {
		local_agg_sizes.push_back(a.size());
	}

	auto aggsizes = flecsi::util::mpi::all_gatherv(local_agg_sizes, comm);

	flecsi::util::offsets row_part, proc_part;
	for (const auto & proc_sizes : aggsizes) {
		for (const auto & color_size : proc_sizes) {
			row_part.push_back(color_size);
		}
		proc_part.push_back(proc_sizes.size());
	}

	return std::make_tuple(proc_part, row_part);
}

template<class scalar, class size>
void aggregate_and_partition(
	float beta,
	csr_acc<scalar, size> AA,
	std::vector<aggregate_t> & agg_out,
	aggt_acc<flecsi::wo, flecsi::na> aggt,
	csr_init<scalar, size> & topo_init) {
	std::vector<aggregate_t> aggs;
	for (auto A : AA.accessors()) {
		aggs.emplace_back(double_pairwise_agg(beta, A.diag(), A.offd(), true));
	}

	auto comm = MPI_COMM_WORLD;
	auto [proc_part, row_part] = create_partition(comm, aggs);

	{ // fill field with transpose of aggregates
		auto agg = aggs.begin();
		for (auto at : aggt.accessors()) {
			std::fill(at.span().begin(),
			          at.span().end(),
			          std::numeric_limits<flecsi::util::id>::max());
			auto offset = row_part(proc_part(flecsi::process()));
			transpose_aggregates(*agg, at, offset);
			++agg;
		}
	}

	topo_init.comm = comm;
	topo_init.row_part.set_offsets(row_part);
	topo_init.col_part.set_offsets(row_part);
	topo_init.proc_part.set_offsets(proc_part);
	topo_init.nrows = row_part.total();
	topo_init.ncols = row_part.total();

	agg_out = std::move(aggs);
}

template<class scalar, class size>
void coarsen_with_aggregates(
	csr_acc<scalar, size> AA,
	const std::vector<aggregate_t> & aggs,
	aggt_acc<flecsi::ro, flecsi::ro> aggts,
	csr_init<scalar, size> & topo_init) {

	auto agg_it = aggs.begin();
	auto aggt_it = aggts.accessors().begin();
	for (auto A : AA.accessors()) {
		const auto & agg = *agg_it++;
		auto aggt = *aggt_it++;
		mat::csr<scalar, size> loc{agg.size(), agg.size()};

		auto [rowptr_coarse, colind_coarse, values_coarse] = loc.data.vecs();
		for (size rc = 0; rc < agg.size(); ++rc) {
			// coarse column index -> aggregated value
			std::map<size, scalar> agg_values;
			for (auto r : agg[rc]) {
				auto collapse = [&](const auto & mat) {
					auto [rowptr, colind, values] = mat.rep();
					for (size off = rowptr[r]; off < rowptr[r + 1]; ++off) {
						if (aggt[colind[off]] !=
						    std::numeric_limits<flecsi::util::id>::max())
							agg_values[aggt[colind[off]]] += values[off];
					}
				};
				collapse(A.diag());
				collapse(A.offd());
			}

			size off{0};
			for (const auto & ce : agg_values) {
				colind_coarse.push_back(ce.first);
				values_coarse.push_back(ce.second);
				++off;
			}
			rowptr_coarse[rc + 1] = rowptr_coarse[rc] + off;
		}

		topo_init.proc_mats.push_back(std::move(loc));
	}
}

} // namespace task

template<class scalar, class size, class Ref>
auto coarsen(const mat::parcsr<scalar, size> & Af, Ref aggt, float beta = 0.25) {
	typename topo::csr<scalar, size>::init topo_init;
	std::vector<task::aggregate_t> agg;
	auto lm = flecsi::data::launch::make(Af.data.topo());
	flecsi::execute<task::aggregate_and_partition<scalar, size>, flecsi::mpi>(
		beta, lm, agg, aggt, topo_init);
	flecsi::execute<task::coarsen_with_aggregates<scalar, size>, flecsi::mpi>(
		lm, agg, aggt, topo_init);

	return mat::parcsr<scalar, size>(std::move(topo_init));
}

}

#endif
