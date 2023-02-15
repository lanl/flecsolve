#include <flecsi/topo/narray/interface.hh>

namespace flecsolve {

struct testmesh : flecsi::topo::specialization<flecsi::topo::narray, testmesh> {
	enum index_space { cells };
	using index_spaces = has<cells>;
	// enum domain { logical, all, global };
	enum axis { x_axis };
	using axes = has<x_axis>;
	// enum boundary { low, high };
	using coord = base::coord;
	using gcoord = base::gcoord;
	using colors = base::colors;
	using hypercube = base::hypercube;
	using axis_definition = base::axis_definition;
	using index_definition = base::index_definition;
	using domain = base::domain;

	struct meta_data {};

	static constexpr std::size_t dimension = 1;

	template<auto>
	static constexpr std::size_t privilege_count = 2;
	template<class B>
	struct interface : B {

		template<axis A, domain DM = domain::logical>
		std::size_t size() {
			return B::template size<index_space::cells, A, DM>();
		}
		// 	if constexpr (DM == domain::logical) {
		// 		return B::
		// 			template size<index_space::cells, A, domain::logical>();
		// 	}
		// 	else if (DM == domain::all) {
		// 		return B::
		// 			template size<index_space::cells, A, domain::all>();
		// 	}
		// 	else if (DM == domain::global) {
		// 		return B::
		// 			template size<index_space::cells, A, domain::global>();
		// 	}
		// }

		std::size_t global_id(std::size_t i) const {
			return B::template global_id<index_space::cells, testmesh::x_axis>(
				i);
		}

		template<index_space Space>
		auto dofs() {
			const std::size_t start =
				B::template offset<Space, x_axis, domain::logical>();
			const std::size_t end =
				B::template offset<Space, x_axis, domain::boundary_high>();

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
