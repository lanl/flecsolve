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
#ifndef FLECSOLVE_EXAMPLES_HEAT_MESH_H
#define FLECSOLVE_EXAMPLES_HEAT_MESH_H

#include "flecsi/data.hh"
#include "flecsi/topo/narray/interface.hh"

#include "index_util.hh"

namespace heat {

static constexpr flecsi::privilege na = flecsi::na;
static constexpr flecsi::privilege ro = flecsi::ro;
static constexpr flecsi::privilege wo = flecsi::wo;
static constexpr flecsi::privilege rw = flecsi::rw;

template<typename T, flecsi::data::layout L = flecsi::data::layout::dense>
using field = flecsi::field<T, L>;

struct mesh : flecsi::topo::specialization<flecsi::topo::narray, mesh> {

	enum index_space { vertices };
	using index_spaces = has<vertices>;
	enum domain { interior, extended, all };
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

		template<axis A>
		auto get_axis() const { return B::template axis<mesh::vertices, A>(); }

		template<axis A>
		FLECSI_INLINE_TARGET std::size_t global_id(std::size_t i) const {
			return get_axis<A>().global_id(i);
		}

		template<axis A, domain DM = interior>
		FLECSI_INLINE_TARGET auto vertices() const {
			if constexpr (DM == interior) {
				return flecsi::topo::make_ids<mesh::vertices>(
					get_axis<A>().layout.logical());
			}
			else if constexpr (DM == extended) {
				return flecsi::topo::make_ids<mesh::vertices>(
					get_axis<A>().layout.extended());
			} else if constexpr (DM == all) {
				return flecsi::topo::make_ids<mesh::vertices>(
					get_axis<A>().layout.all());
			}

		}

		void set_geometry(double x, double y) { // available if writable
			this->policy_meta() = {x, y};
		}

		double xdelta() { return this->policy_meta().xdelta; }

		double ydelta() { return this->policy_meta().ydelta; }

		double dxdy() { return xdelta() * ydelta(); }

		template<axis A>
		double value(std::size_t i) {
			return (A == x_axis ? xdelta() : ydelta()) * global_id<A>(i);
		}

		template<axis A, boundary BD>
		bool is_boundary(std::size_t i) {
			const flecsi::topo::narray_impl::axis_info & a = get_axis<A>();
			const bool l = a.low();
			const bool h = a.high();

			const auto loff = a.layout.logical<0>();
			const auto lsize = a.layout.logical<1>() - loff;

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
			const auto & l = get_axis<A>().layout;

			return {l.template logical<0>(), l.template logical<1>()};
		}

		template<index_space Space, auto... Axis>
		auto interior_subrange(flecsi::util::constants<Axis...>) {
			return std::array<util::srange, sizeof...(Axis)>{
				{interior_subrange<Space, Axis>()...}};
		}

		template<index_space Space, auto... Axis>
		auto extents_array(flecsi::util::constants<Axis...>) {
			return std::array<std::size_t, sizeof...(Axis)>{
				get_axis<Axis>().layout.extent()...};
		}
	};

	static coloring color(const index_definition & idef) { return {{idef}}; }

	using grect = std::array<std::array<double, 2>, 2>;

	static void set_geometry(mesh::accessor<flecsi::rw> sm, grect const & g) {
		sm.set_geometry(
			std::abs(g[0][1] - g[0][0]) / (sm.get_axis<x_axis>().layout.extent() - 1),
			std::abs(g[1][1] - g[1][0]) / (sm.get_axis<y_axis>().layout.extent() - 1));
	}

	static void initialize(flecsi::scheduler & s,
	                       mesh::topology & topo,
	                       coloring const &,
	                       grect const & geometry) {
		flecsi::execute<set_geometry, flecsi::mpi>(topo, geometry);
	}
};

inline std::array<field<double>::definition<mesh, mesh::vertices>, 2> ud;

}

#endif
