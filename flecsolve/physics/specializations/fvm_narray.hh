#pragma once

#include <bits/utility.h>
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
// using field = vec::data::field<T, L>;
using field = flecsi::field<T, L>;

namespace fvmtools {

constexpr std::array<std::size_t, 3> idx {0, 1, 2};
constexpr std::array<std::size_t, 3> idy {2, 0, 1};
constexpr std::array<std::size_t, 3> idz {1, 2, 0};
//constexpr std::array<std::array<std::size_t, 3>, 3> idv {xr, yr, zr};
constexpr flecsi::util::key_array<std::array<std::size_t, dimension>, axes> idv {xr, yr, zr};

inline static std::size_t digit(flecsi::util::id & x, std::size_t d) {
	std::size_t ret = x % d;
	x /= d;
	return ret;
}

template<class V>
inline flecsi::util::id
translate(flecsi::util::id & x, std::size_t stride, const V & sub) {
	flecsi::util::id ret;

	ret = digit(x, sub.size()) + static_cast<std::size_t>(sub.front());

	ret *= stride;

	return ret;
}
template<class... Vs, std::size_t... Index>
std::size_t subranges_size(std::tuple<Vs...> & subranges,
                           std::index_sequence<Index...>) {
	return (std::get<Index>(subranges).size() * ...);
}

template<class... Vs, std::size_t... Index>
flecsi::util::id
translate_index(flecsi::util::id x,
                const std::tuple<Vs...> & subranges,
                const std::array<std::size_t, sizeof...(Vs)> & strides,
                std::index_sequence<Index...>) {
	return (translate(x, strides[Index], std::get<Index>(subranges)) + ...);
}

template<auto Axis, std::size_t I, std::size_t Dim>
static constexpr auto jump_idx() {
	return static_cast<decltype(Axis)>((Axis + I) % Dim);
}

template<auto Axis, std::size_t Dim>
static constexpr auto next_idx() {
	return jump_idx<Axis, 1, Dim>();
}

}

struct fvm_narray
	: flecsi::topo::specialization<flecsi::topo::narray, fvm_narray> {
	enum index_space { cells, faces };
	using index_spaces = has<cells, faces>;

	enum domain {
		logical,
		extended,
		all,
		boundary_low,
		boundary_high,
		ghost_low,
		ghost_high,
		global
	};

	enum axis { x_axis, y_axis, z_axis };
	using axes = has<x_axis, y_axis, z_axis>;

	using coord = base::coord;
	using colors = base::colors;
	using hypercube = base::hypercube;
	using coloring_definition = base::coloring_definition;

	static constexpr std::size_t dimension = 3;

	struct meta_data {
		flecsi::util::key_array<double, axes> delta;
	};

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
		static constexpr decltype(auto) NAD() {
			if constexpr (DM == logical)
				return B::domain::logical;
			else if (DM == extended)
				return B::domain::extended;
			else if (DM == all)
				return B::domain::all;
			else if (DM == boundary_low)
				return B::domain::boundary_low;
			else if (DM == boundary_high)
				return B::domain::boundary_high;
			else if (DM == ghost_low)
				return B::domain::ghost_low;
			else if (DM == ghost_high)
				return B::domain::ghost_high;
			else if (DM == global)
				return B::domain::global;
		}

		template<index_space IS, axis A>
		std::size_t extents() const {
			return B::template extents<IS, A>();
		}
		template<index_space IS, axis A, domain DM = logical>
		std::size_t size() const {
			return B::template size<IS, A, NAD<DM>()>();
		}

		template<index_space IS, axis A, domain DM = logical>
		constexpr auto range() {
			return B::template range<IS, A, NAD<DM>()>();
		}

		template<index_space INIS, axis INAX, domain DM = logical>
		constexpr auto full_range() {
			using namespace fvmtools;

			return std::make_tuple(
				range<INIS, INAX, DM>(),
				range<cells, jump_idx<INAX, 1, dimension>(), logical>(),
				range<cells, jump_idx<INAX, 2, dimension>(), logical>());
		}

		template<index_space INIS, axis INAX, domain DM = logical>
		constexpr auto full_range_flat() {
			using namespace fvmtools;
			auto stacked = full_range<INIS, INAX, DM>();
			auto strides = utils::make_array(
				extents<INIS, INAX>() / extents<INIS, INAX>(),
				extents<INIS, INAX>(),
				extents<cells, jump_idx<INAX, 1, dimension>()>() *
					extents<INIS, INAX>());
			return flecsi::util::transform_view(
				flecsi::util::iota_view<flecsi::util::id>(
					0,
					subranges_size(stacked,
			                       std::make_index_sequence<dimension>())),
				[=](const auto & x) {
					return flecsi::topo::id<INIS>(
						translate_index(x,
				                        stacked,
				                        strides,
				                        std::make_index_sequence<dimension>()));
				});
		}

		template<axis INAX = x_axis,
		         index_space INIS = cells,
		         typename T,
		         flecsi::Privileges P>
		FLECSI_INLINE_TARGET auto mdspanx(
			flecsi::data::accessor<flecsi::data::dense, T, P> const & a) const {
			using namespace fvmtools;
			auto const s = a.span();
			auto const exs = utils::make_array(
				extents<INIS, INAX>(),
				extents<cells, jump_idx<INAX, 1, dimension>()>(),
				extents<cells, jump_idx<INAX, 2, dimension>()>());
			return flecsi::util::mdspan<typename decltype(s)::element_type,
			                            dimension>(s.data(), exs);
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
		constexpr auto dofs() {
			return full_range_flat<Space, x_axis>();
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
			ne, nc, pcs, aux_ex, MPI_COMM_WORLD, false, false);

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
	static auto set_delta(fvm_narray::accessor<flecsi::rw> sm, const gbox & g) {
		if (sm.size<fvm_narray::cells, A, fvm_narray::global>() <= 1) {
			return 1.0;
		}
		return std::abs(g[A][1] - g[A][0]) /
		       (sm.size<fvm_narray::cells, A, fvm_narray::global>() - 1);
	}

	template<auto... Axis>
	static flecsi::util::key_array<double, axes>
	geom(fvm_narray::accessor<flecsi::rw> sm, const gbox & g, has<Axis...>) {
		return {set_delta<Axis>(sm, g)...};
	}

	static void set_geometry(fvm_narray::accessor<flecsi::rw> sm,
	                         gbox const & g) {
		meta_data & md = sm.policy_meta_;
		md.delta = geom(sm, g, axes());

	}

	static void initialize(flecsi::data::topology_slot<fvm_narray> & s,
	                       coloring const &,
	                       gbox const & geometry) {
		flecsi::execute<set_geometry, flecsi::mpi>(s, geometry);
	} // initialize
};

}
}
