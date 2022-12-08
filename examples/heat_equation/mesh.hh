#ifndef FLECSOLVE_EXAMPLES_HEAT_MESH_H
#define FLECSOLVE_EXAMPLES_HEAT_MESH_H

#include "flecsi/data.hh"
#include "flecsi/topo/narray/interface.hh"

#include "index_util.hh"

namespace heat {

static constexpr flecsi::partition_privilege_t na = flecsi::na;
static constexpr flecsi::partition_privilege_t ro = flecsi::ro;
static constexpr flecsi::partition_privilege_t wo = flecsi::wo;
static constexpr flecsi::partition_privilege_t rw = flecsi::rw;

template<typename T, flecsi::data::layout L = flecsi::data::layout::dense>
using field = flecsi::field<T, L>;

struct mesh : flecsi::topo::specialization<flecsi::topo::narray, mesh> {

	enum index_space { vertices };
	using index_spaces = has<vertices>;
	enum domain { interior, logical, all, global };
	enum axis { x_axis, y_axis };
	using axes = has<x_axis, y_axis>;
	enum boundary { low, high };

	using coord = base::coord;
	using gcoord = base::gcoord;
	using colors = base::colors;
	using hypercube = base::hypercube;
	using axis_definition = base::axis_definition;
	using index_definition = base::index_definition;

	struct meta_data {
		double xdelta;
		double ydelta;
	};

	static constexpr std::size_t dimension = 2;

	template<auto>
	static constexpr std::size_t privilege_count = 2;

	/*--------------------------------------------------------------------------*
	  Interface.
	  *--------------------------------------------------------------------------*/

	template<class B>
	struct interface : B {

		template<axis A, domain DM = interior>
		std::size_t size() {
			if constexpr (DM == interior) {
				const bool low = B::template is_low<index_space::vertices, A>();
				const bool high =
					B::template is_high<index_space::vertices, A>();

				if (low && high) { /* degenerate */
					return size<A, logical>() - 2;
				}
				else if (low || high) {
					return size<A, logical>() - 1;
				}
				else { /* interior */
					return size<A, logical>();
				}
			}
			else if constexpr (DM == logical) {
				return B::template size<index_space::vertices,
				                        A,
				                        base::domain::logical>();
			}
			else if (DM == all) {
				return B::template size<index_space::vertices,
				                        A,
				                        base::domain::all>();
			}
			else if (DM == global) {
				return B::template size<index_space::vertices,
				                        A,
				                        base::domain::global>();
			}
		}

		template<axis A>
		FLECSI_INLINE_TARGET std::size_t global_id(std::size_t i) const {
			return i - B::template logical<mesh::vertices, A, 0>() +
			       B::template offset<mesh::vertices,
			                          A,
			                          base::domain::global>();
		}

		template<axis A, domain DM = interior>
		auto vertices() {
			if constexpr (DM == interior) {
				const bool low = B::template is_low<index_space::vertices, A>();
				const bool high =
					B::template is_high<index_space::vertices, A>();
				const std::size_t start =
					B::template logical<index_space::vertices, A, 0>();
				const std::size_t end =
					B::template logical<index_space::vertices, A, 1>();

				return flecsi::topo::make_ids<index_space::vertices>(
					flecsi::util::iota_view<flecsi::util::id>(start + low,
				                                              end - high));
			}
			else if constexpr (DM == logical) {
				return B::template range<index_space::vertices,
				                         A,
				                         base::domain::logical>();
			}
			else if (DM == all) {
				return B::template range<index_space::vertices,
				                         A,
				                         base::domain::all>();
			}
		}

		double xdelta() { return (*(this->policy_meta_)).xdelta; }

		double ydelta() { return (*(this->policy_meta_)).ydelta; }

		double dxdy() { return xdelta() * ydelta(); }

		template<axis A>
		double value(std::size_t i) {
			return (A == x_axis ? xdelta() : ydelta()) * global_id<A>(i);
		}

		template<axis A, boundary BD>
		bool is_boundary(std::size_t i) {

			auto const loff = B::template offset<index_space::vertices,
			                                     A,
			                                     base::domain::logical>();
			auto const lsize = B::template size<index_space::vertices,
			                                    A,
			                                    base::domain::logical>();
			const bool l = B::template is_low<index_space::vertices, A>();
			const bool h = B::template is_high<index_space::vertices, A>();

			if (l && h) { /* degenerate */
				if constexpr (BD == boundary::low) {
					return i == loff;
				}
				else {
					return i == (lsize + loff - 1);
				}
			}
			else if (l) {
				if constexpr (BD == boundary::low) {
					return i == loff;
				}
				else {
					return false;
				}
			}
			else if (h) {
				if constexpr (BD == boundary::low) {
					return false;
				}
				else {
					return i == (lsize + loff - 1);
				}
			}
			else { /* interior */
				return false;
			}
		}

		template<index_space Space>
		auto dofs() {
			return util::make_subrange_ids<Space, axes::size>(
				interior_subrange<Space>(axes()), extents_array<Space>(axes()));
		}

	protected:
		template<index_space Space, axis A>
		util::srange interior_subrange() {
			const bool low = B::template is_low<Space, A>();
			const bool high = B::template is_high<Space, A>();
			const std::size_t start = B::template logical<Space, A, 0>();
			const std::size_t end = B::template logical<Space, A, 1>();
			return {start + low, end - high};
		}

		template<index_space Space, auto... Axis>
		auto interior_subrange(flecsi::util::constants<Axis...>) {
			return std::array<util::srange, sizeof...(Axis)>{
				{interior_subrange<Space, Axis>()...}};
		}

		template<index_space Space, auto... Axis>
		auto extents_array(flecsi::util::constants<Axis...>) {
			return std::array<std::size_t, sizeof...(Axis)>{
				{B::template extents<Space>().template get<Axis>()...}};
		}
	};

	static coloring color(std::size_t num_colors, gcoord axis_extents) {
		index_definition idef;
		idef.axes =
			flecsi::topo::narray_utils::make_axes(num_colors, axis_extents);
		for (auto & a : idef.axes) {
			a.hdepth = 1;
		}

		flog_assert(idef.colors() == flecsi::processes(),
		            "current implementation is restricted to 1-to-1 mapping");

		return {MPI_COMM_WORLD, {idef}};
	}

	using grect = std::array<std::array<double, 2>, 2>;

	static void set_geometry(mesh::accessor<flecsi::rw> sm, grect const & g) {
		meta_data & md = sm.policy_meta_;
		double xdelta =
			std::abs(g[0][1] - g[0][0]) / (sm.size<x_axis, global>() - 1);
		double ydelta =
			std::abs(g[1][1] - g[1][0]) / (sm.size<y_axis, global>() - 1);

		md.xdelta = xdelta;
		md.ydelta = ydelta;
	}

	static void initialize(flecsi::data::topology_slot<mesh> & s,
	                       coloring const &,
	                       grect const & geometry) {
		flecsi::execute<set_geometry, flecsi::mpi>(s, geometry);
	}
};

inline mesh::slot m;
inline mesh::cslot coloring;

inline std::array<field<double>::definition<mesh, mesh::vertices>, 2> ud;

}

#endif
