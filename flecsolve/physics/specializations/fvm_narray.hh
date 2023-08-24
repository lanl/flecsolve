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

// TODO: seperate from topology
namespace fvmtools {

template<std::size_t Dim>
constexpr auto makeis() {
	return std::make_index_sequence<Dim>();
}

// axis rotation helpers
constexpr std::array<std::size_t, 3> idx{0, 1, 2};
constexpr std::array<std::size_t, 3> idy{2, 0, 1};
constexpr std::array<std::size_t, 3> idz{1, 2, 0};

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

// utilities for doing arithmatic on `axes`
template<auto Axis, std::size_t I, std::size_t Dim>
static constexpr auto jump_idx() {
	return static_cast<decltype(Axis)>((Axis + I) % Dim);
}

template<auto Axis, std::size_t Dim>
static constexpr auto next_idx() {
	return jump_idx<Axis, 1, Dim>();
}

// utilitiy for `if constexpr` no-match case
template<bool flag = false>
void static_no_match() {
	static_assert(flag, "no match");
}

/**
 * @brief applys a function on a set of cells defined by a tuple of ranges
 *
 * @tparam Box a 3-D view of the data
 * @tparam IdxT a tuple of index ranges, ordered `z`,`y`,`x` (though not
 * required)
 * @tparam F the function to apply
 * @tparam Args arguments to the function
 */
template<class Box, class IdxT, class F, class... Args>
inline void apply_to(Box box, IdxT && idxs, F && f, Args &&... args) {
	for (auto k : std::get<0>(idxs)) {
		for (auto j : std::get<1>(idxs)) {
			for (auto i : std::get<2>(idxs)) {
				box[k][j][i] = f(std::forward<Args>(args)...);
			}
		}
	}
}

/**
 * @brief same as `apply_to`, but explicitly passes the indicies to the function
 *
 * @tparam Box a 3-D view of the data
 * @tparam IdxT a tuple of index ranges, ordered `z`,`y`,`x` (though not
 * required)
 * @tparam F the function to apply
 * @tparam Args arguments to the function
 */
template<class Box, class IdxT, class F, class... Args>
inline void
apply_to_with_index(Box box, IdxT && idxs, F && f, Args &&... args) {
	for (auto k : std::get<0>(idxs)) {
		for (auto j : std::get<1>(idxs)) {
			for (auto i : std::get<2>(idxs)) {
				box[k][j][i] = f(k, j, i, std::forward<Args>(args)...);
			}
		}
	}
}

}

struct fvm_narray
	: flecsi::topo::specialization<flecsi::topo::narray, fvm_narray> {
	using base = typename fvm_narray::base;
	enum index_space { cells, faces };
	using index_spaces = has<cells, faces>;

	enum axis { x_axis, y_axis, z_axis };
	using axes = has<x_axis, y_axis, z_axis>;

	using coord = typename base::coord;
	using domain = typename base::domain;
	using gcoord = typename base::gcoord;
	using colors = typename base::colors;
	using coloring = typename base::coloring;
	using hypercube = typename base::hypercube;
	using axis_definition = typename base::axis_definition;
	using index_definition = typename base::index_definition;

	static constexpr std::size_t dimension = 3;

	struct meta_data {
		flecsi::util::key_array<double, axes> delta;
	};

	template<auto>
	static constexpr std::size_t privilege_count = 2;

	template<class B>
	struct interface : B {

		template<index_space IS, axis A, domain DM = base::domain::logical>
		std::size_t size() const {
			return B::template size<IS, A, DM>();
		}

		template<index_space IS, axis A, domain DM = base::domain::logical>
		constexpr auto range() {
			return B::template range<IS, A, DM>();
		}

		template<index_space MAJORSPACE = cells,
		         axis MAJORAXIS = x_axis,
		         domain DM = base::domain::logical>
		constexpr auto full_range() {
			using namespace fvmtools;
			if constexpr (MAJORAXIS == x_axis) {
				return std::make_tuple(
					range<cells, z_axis, base::domain::logical>(),
					range<cells, y_axis, base::domain::logical>(),
					range<MAJORSPACE, x_axis, DM>());
			}
			else if constexpr (MAJORAXIS == y_axis) {
				return std::make_tuple(
					range<cells, z_axis, base::domain::logical>(),
					range<MAJORSPACE, y_axis, DM>(),
					range<cells, x_axis, base::domain::logical>());
			}
			else if constexpr (MAJORAXIS == z_axis) {
				return std::make_tuple(
					range<MAJORSPACE, z_axis, DM>(),
					range<cells, y_axis, base::domain::logical>(),
					range<cells, x_axis, base::domain::logical>());
			}
			else {
				static_no_match();
			}
		}

		template<axis A, index_space S = cells>
		FLECSI_INLINE_TARGET std::size_t global_id(std::size_t i) const {
			return i - B::template offset<S, A, base::domain::logical>() +
			       B::template offset<S, A, base::domain::global>();
		}

		template<axis A>
		static constexpr std::uint32_t to_idx() {
			return B::template to_idx<A>();
		}

		template<index_space Space>
		constexpr auto dofs() {
			using namespace fvmtools;
			auto stacked =
				std::make_tuple(range<cells, x_axis, base::domain::logical>(),
			                    range<cells, y_axis, base::domain::logical>(),
			                    range<cells, z_axis, base::domain::logical>());
			auto strides =
				utils::make_array(size<cells, x_axis, base::domain::all>() /
			                          size<cells, x_axis, base::domain::all>(),
			                      size<cells, x_axis, base::domain::all>(),
			                      size<cells, x_axis, base::domain::all>() *
			                          size<cells, y_axis, base::domain::all>());

			return flecsi::util::transform_view(
				flecsi::util::iota_view<flecsi::util::id>(
					0, subranges_size(stacked, makeis<dimension>())),
				[=](const auto & x) {
					return flecsi::topo::id<cells>(translate_index(
						x, stacked, strides, makeis<dimension>()));
				});
		}

		template<axis A>
		double dx() {
			return this->policy_meta().delta[A];
		}

		template<axis A>
		double value(std::size_t i) {
			return (dx<A>() *
			        (static_cast<double>(
						global_id<A>(i) +
						B::template size<index_space::cells,
			                             A,
			                             base::domain::ghost_low>())));
		}

		double volume() { return product(axes()); }

		template<axis A>
		double normal_dA() {
			using nrm = utils::mp::complement_list<has<A>, axes>;
			return product(nrm());
		}

		template<axis... As>
		constexpr auto product(has<As...>) {
			return (dx<As>() * ... * 1.0);
		}

		template<class Box, axis... Ax>
		constexpr flecsi::util::key_array<double, axes>
		box_geom(const Box & box, has<Ax...>) {
			return {{std::abs(box[Ax][1] - box[Ax][0]) /
			         (size<cells, Ax, domain::all>() - 1)...}};
		}

		template<class Box>
		constexpr auto set_geom(const Box & box) {
			this->policy_meta() = {box_geom(box, axes())};
		}

	}; // interface

	static auto distribute(std::size_t np, std::vector<std::size_t> indices) {
		return flecsi::topo::narray_utils::distribute(np, indices);
	}

	static coloring color(std::size_t num_colors, gcoord axis_extents) {
		index_definition idef;
		idef.axes =
			flecsi::topo::narray_utils::make_axes(num_colors, axis_extents);
		for (auto & a : idef.axes) {
			a.hdepth = 1;
			a.bdepth = 1;
		}

		flog_assert(idef.colors() == flecsi::processes(),
		            "current implementation is restricted to 1-to-1 mapping");

		index_definition idef_faces;
		idef_faces.axes =
			flecsi::topo::narray_utils::make_axes(num_colors, axis_extents);

		for (auto & a : idef_faces.axes) {
			a.hdepth = 0;
			a.bdepth = 0;
			a.auxiliary = true;
		}

		flog_assert(idef_faces.colors() == flecsi::processes(),
		            "current implementation is restricted to 1-to-1 mapping");

		return {{idef, idef_faces}};
	}

	using gbox = flecsi::util::key_array<std::array<double, 2>, axes>;

	static void set_geometry(fvm_narray::accessor<flecsi::rw> sm,
	                         gbox const & g) {
		sm.set_geom(g);
	}

	static void initialize(flecsi::data::topology_slot<fvm_narray> & s,
	                       coloring const &,
	                       gbox const & geometry) {

		flecsi::execute<set_geometry, flecsi::mpi>(s, geometry);
	} // initialize
};

}
}
