#include <flecsi/topo/narray/interface.hh>

namespace flecsolve {

struct testmesh : flecsi::topo::specialization<flecsi::topo::narray, testmesh> {
	enum index_space { cells };
	using index_spaces = has<cells>;
	enum domain { logical, all, global };
	enum axis { x_axis };
	using axes = has<x_axis>;
	enum boundary { low, high };
	using coord = base::coord;
	using gcoord = base::gcoord;
	using colors = base::colors;
	using hypercube = base::hypercube;
	using axis_definition = base::axis_definition;
	using index_definition = base::index_definition;

	struct meta_data {
	};

	static constexpr std::size_t dimension = 1;

	template<auto>
	static constexpr std::size_t privilege_count = 2;
	template<class B>
	struct interface : B {

		template<axis A, domain DM = logical>
		std::size_t size() {
			if constexpr (DM == logical) {
				return B::template size<cells, x_axis, base::domain::logical>();
			}
			else if (DM == all) {
				return B::template size<cells, x_axis, base::domain::all>();
			}
			else if (DM == global) {
				return B::template size<cells, x_axis, base::domain::global>();
			}
		}

		FLECSI_INLINE_TARGET std::size_t global_id(std::size_t i) const {
			return i -
			       B::template offset<cells, x_axis, base::domain::logical>() +
			       B::template offset<cells, x_axis, base::domain::global>();
		}

		template<index_space Space>
		auto dofs() {
			const std::size_t start =
				B::template offset<Space, x_axis, base::domain::logical>();
			const std::size_t end = B::
				template offset<Space, x_axis, base::domain::boundary_high>();

			return flecsi::topo::make_ids<Space>(
				flecsi::util::iota_view<flecsi::util::id>(start, end));
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

	static void initialize(flecsi::data::topology_slot<testmesh> &,
	                       coloring const &) {}
};

}
