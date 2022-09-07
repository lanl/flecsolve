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
	using colors = base::colors;
	using hypercube = base::hypercube;
	using coloring_definition = base::coloring_definition;

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
				return B::
					template size<index_space::cells, A, B::domain::logical>();
			}
			else if (DM == all) {
				return B::
					template size<index_space::cells, A, B::domain::all>();
			}
			else if (DM == global) {
				return B::
					template size<index_space::cells, A, B::domain::global>();
			}
		}

		std::size_t global_id(std::size_t i) const {
			return B::template global_id<index_space::cells, testmesh::x_axis>(
				i);
		}

		template<index_space Space>
		auto dofs() {
			const std::size_t start = B::template logical<Space, x_axis, 0>();
			const std::size_t end = B::template logical<Space, x_axis, 1>();

			return flecsi::topo::make_ids<Space>(
				flecsi::util::iota_view<flecsi::util::id>(start, end));
		}
	};

	static auto distribute(std::size_t np, std::vector<std::size_t> indices) {
		return flecsi::topo::narray_utils::distribute(np, indices);
	}

	static coloring color(colors axis_colors, coord axis_extents) {
		coord hdepths{1};
		coord bdepths{0};
		std::vector<bool> periodic{false};
		coloring_definition cd{
			axis_colors, axis_extents, hdepths, bdepths, periodic};

		auto [nc, ne, pcs, partitions] =
			flecsi::topo::narray_utils::color(cd, MPI_COMM_WORLD);

		coloring c;
		c.comm = MPI_COMM_WORLD;
		c.colors = nc;
		c.idx_colorings.emplace_back(std::move(pcs));
		c.partitions.emplace_back(std::move(partitions));
		return c;
	}

	static void initialize(flecsi::data::topology_slot<testmesh> &,
	                       coloring const &) {}
};

}
