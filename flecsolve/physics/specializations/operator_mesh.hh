#pragma once

#include <flecsi/data.hh>
#include <flecsi/topo/narray/coloring_utils.hh>
#include <flecsi/topo/narray/interface.hh>
#include <flecsi/topo/narray/types.hh>
#include <flecsi/util/constant.hh>
#include <tuple>
#include <utility>

#include "flecsolve/physics/common/operator_utils.hh"

namespace flecsolve {
namespace physics {

template<typename T, flecsi::data::layout L>
using field = vec::data::field<T,L>;
//using field = flecsi::field<T, L>;

struct operator_mesh
	: flecsi::topo::specialization<flecsi::topo::narray, operator_mesh> {
	enum index_space { cells, faces };
	using index_spaces = has<cells, faces>;

	enum domain { logical, extended, all, global, boundary_low, boundary_high };

	enum axis { x_axis, y_axis, z_axis };
	using axes = has<x_axis, y_axis, z_axis>;

	using coord = base::coord;
	using colors = base::colors;
	using hypercube = base::hypercube;
	using coloring_definition = base::coloring_definition;

	struct meta_data {
		flecsi::util::key_array<double, axes> delta;
	};

	static constexpr std::size_t dimension = 3;

	template<auto>
	static constexpr std::size_t privilege_count = 2;

	template<class B>
	struct interface : B {
		/**
		 * @brief Currently, the domain enum of the topology is not mapped into
		 * the interface, this is workaround
		 *
		 * @tparam DM interface domain
		 * @return constexpr decltype(auto) topo domain
		 */
		template<domain DM>
		static constexpr decltype(auto) __dm() {
			if constexpr (DM == logical)
				return B::domain::logical;
			else if (DM == extended)
				return B::domain::extended;
			else if (DM == all)
				return B::domain::all;
			else if (DM == global)
				return B::domain::global;
			else if (DM == boundary_low)
				return B::domain::boundary_low;
			else if (DM == boundary_high)
				return B::domain::boundary_high;
		}

		template<index_space IS, axis A, domain DM = logical>
		auto range() {
			return B::template range<IS, A, __dm<DM>()>();
		}

		template<index_space IS, axis A, domain DM = logical>
		std::size_t size() const {
			return B::template size<IS, A, __dm<DM>()>();
		}

		template<axis A, index_space IS = cells>
		std::size_t global_id(std::size_t i) const {
			return B::template global_id<IS, A>(i);
		}

		template<axis A>
		static constexpr std::uint32_t to_idx() {
			return B::template to_idx<A>();
		}

		template<index_space Space>
		auto dofs() {
			return get_stencil<x_axis, Space>(utils::offset_seq<>());
		}

		template<axis A>
		double dx() {
			return (*(this->policy_meta_)).delta[A];
		}

		template<axis A>
		double value(std::size_t i) {
			return (dx<A>() * (static_cast<double>(
								  global_id<A>(i) +
								  B::template size<index_space::cells,
			                                       A,
			                                       B::domain::ghost_low>())));
		}

		template<axis A>
		double normal_dA() {
			using nrm = utils::mp::complement_list<has<A>, axes>;
			return product(nrm());
		}

		template<axis... As>
		constexpr auto product(has<As...>) {
			return (dx<As>() * ... * 1.0);
		}

		/**
		 * @brief returns a tuple of an index iterator that is continuous along
		 * the specified axis, and a set of offset values based on the requested
		 * stencil values. "stenils" here refers to a set of integer literals.
		 *
		 * @tparam Along axis of iteration
		 * @tparam From index space of domain indicies
		 * @tparam To index space of range indicies
		 * @tparam DM domain of indices, e.g. boundary or interior
		 * @tparam II integer sequence of "stencil" values along axis A. e.g.
		 * {-1, 1} for left, right
		 * @return decltype(auto)
		 */
		template<axis Along,
		         index_space From,
		         index_space To = From,
		         domain DM = logical,
		         int... II>
		decltype(auto) get_stencil(std::integer_sequence<int, II...>) {
			// get an axis list "rotated" into the axis to get indicies
			using xx = utils::mp::rotate_to<Along, axes>;

			// subrange (begin, end) indicies to get
			auto sr = grid_subrange<index_space::cells, DM, Along>(xx());
			// strides of memory space
			auto st = strides_array<index_space::cells>(axes());

			// there's probably a better way to do this, but for now...
			if constexpr (To == index_space::faces) {
				sr[0].end += 1;
			}

			// we want to step through axis Along, so rotate strides so that
			// these indicies are "continuous" in that direction NOTE: we do
			// this based on how the indicies are calculated, see
			// make_subrange_ids
			std::rotate(st.begin(), st.begin() + to_idx<Along>(), st.end());

			// if the "stecil" is not provided, just return the iterator
			if constexpr (sizeof...(II) == 0) {
				return utils::make_subrange_ids<To, axes::size>(sr, st);
			}
			// otherwise, return a tuple with:
			// get<0>: the iterator
			// get<N > 0>: the offset values for the stenil.
			else {
				return std::make_tuple(
					utils::make_subrange_ids<To, axes::size>(sr, st),
					(static_cast<int>(st[0]) * II)...);
			}
		}

	protected:
		template<index_space Space, domain DM, axis Along, axis A>
		utils::srange grid_subrange() {
			if constexpr (Along != A) {
				const std::size_t start =
					B::template logical<index_space::cells, A, 0>();
				const std::size_t end =
					B::template logical<index_space::cells, A, 1>();
				return {start, end};
			}
			else {
				if constexpr (DM == logical) {
					const std::size_t start =
						B::template logical<Space, A, 0>();
					const std::size_t end = B::template logical<Space, A, 1>();
					return {start, end};
				}
				else if constexpr (DM == boundary_low) {
					const std::size_t start = 0;
					const std::size_t end =
						B::template size<Space, A, __dm<DM>()>();
					return {start, end};
				}
				else if constexpr (DM == boundary_high) {
					const std::size_t start =
						B::template logical<Space, A, 1>();
					const std::size_t end =
						start + B::template size<Space, A, __dm<DM>()>();
					return {start, end};
				}
				else {
					static_assert("boundary not given\n");
				}
			}
		}

		template<index_space Space, domain DM, auto Along, auto... Axis>
		auto grid_subrange(flecsi::util::constants<Axis...>) {
			return std::array<utils::srange, sizeof...(Axis)>{
				{grid_subrange<Space, DM, Along, Axis>()...}};
		}

		template<index_space Space, auto... Axis>
		auto grid_subrange(flecsi::util::constants<Axis...>) {
			return grid_subrange<Space, domain::logical, x_axis>(axes());
		}

		template<index_space Space, axis A>
		std::size_t strides_array(std::size_t & stride) {
			std::size_t ret = stride;
			stride *= B::template extents<Space>().template get<A>();
			return ret;
		}

		template<index_space Space, auto... Axis>
		auto strides_array(flecsi::util::constants<Axis...>) {
			std::size_t stride = 1;
			return std::array<std::size_t, sizeof...(Axis)>{
				{strides_array<Space, Axis>(stride)...}};
		}

	}; // interface

	static auto distribute(std::size_t np, std::vector<std::size_t> indices) {
		return flecsi::topo::narray_utils::distribute(np, indices);
	}

	static coloring color(colors axis_colors, coord axis_extents) {
		coord hdepths(dimension, 1);
		coord bdepths(dimension, 1);
		std::vector<bool> periodic(dimension, false);
		std::vector<bool> aux_ex(dimension, true);
		coloring_definition cd{
			axis_colors, axis_extents, hdepths, bdepths, periodic};

		auto [nc, ne, pcs, partitions] =
			flecsi::topo::narray_utils::color(cd, MPI_COMM_WORLD);

		auto [fcs, fpartitions] = flecsi::topo::narray_utils::color_auxiliary(
			ne, nc, pcs, aux_ex, MPI_COMM_WORLD, false, true);

		coloring c;
		c.comm = MPI_COMM_WORLD;
		c.colors = nc;
		c.idx_colorings.emplace_back(std::move(pcs));
		c.partitions.emplace_back(std::move(partitions));
		c.idx_colorings.emplace_back(std::move(fcs));
		c.partitions.emplace_back(std::move(fpartitions));
		return c;
	}
	using gbox = flecsi::util::key_array<std::array<double, 2>, axes>;

	template<axis A>
	static auto set_delta(operator_mesh::accessor<flecsi::rw> sm,
	                      const gbox & g) {
		if (sm.size<operator_mesh::cells, A, operator_mesh::global>() <= 1) {
			return 1.0;
		}
		return std::abs(g[A][1] - g[A][0]) /
		       (sm.size<operator_mesh::cells, A, operator_mesh::global>() - 1);
	}

	// template<auto Axis, std::size_t I>
	// static constexpr auto jump_idx() {
	// 	return static_cast<axis>((Axis + I) % dim);
	// }

	// template<axis A>
	// static auto set_outerids(operator_mesh::accessor<flecsi::rw> sm)
	// {
	// 	std::vector<std::size_t> oids;
	// 	const auto nsize = sm.size<
	// 	for(auto k : sm.range<operator_mesh::cells, jump_idx<A, 1>,
	// operator_mesh::logical>())
	// 	{
	// 		for(auto j : sm.range<operator_mesh::cells, jump_idx<A, 2>,
	// operator_mesh::logical>())
	// 		{
	// 			oids.push_back()
	// 		}
	// 	}
	// }

	template<auto... Axis>
	static flecsi::util::key_array<double, axes>
	geom(operator_mesh::accessor<flecsi::rw> sm, const gbox & g, has<Axis...>) {
		return {set_delta<Axis>(sm, g)...};
	}

	static void set_geometry(operator_mesh::accessor<flecsi::rw> sm,
	                         gbox const & g) {
		meta_data & md = sm.policy_meta_;
		md.delta = geom(sm, g, axes());
	}

	static void initialize(flecsi::data::topology_slot<operator_mesh> & s,
	                       coloring const &,
	                       gbox const & geometry) {
		flecsi::execute<set_geometry, flecsi::mpi>(s, geometry);
	} // initialize
};

}
}
