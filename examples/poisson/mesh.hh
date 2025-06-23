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
#ifndef FLECSOLVE_EXAMPLES_POISSON_MESH_H
#define FLECSOLVE_EXAMPLES_POISSON_MESH_H

#include "flecsi/data.hh"
#include "flecsi/topo/narray/interface.hh"
#include "flecsi/data/layout.hh"

#include "index_util.hh"

namespace poisson {

enum class five_pt { c, w, s, ndirs };
enum class nine_pt { c, w, s, sw, nw, ndirs };

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
	using grect = std::array<std::array<double, 2>, 2>;

	/*--------------------------------------------------------------------------*
	  Interface.
	  *--------------------------------------------------------------------------*/

	template<class B>
	struct interface : B {

		template<axis A>
		auto get_axis() const {
			return B::template axis<mesh::vertices, A>();
		}

		template<axis A>
		std::size_t global_id(std::size_t i) const {
			const flecsi::topo::narray_impl::axis_info & a = get_axis<A>();
			const auto l0 = a.layout.logical<0>();
			const auto l1 = a.layout.logical<1>();
			if (a.low() && i == l0 - 1) return l0 - 2;
			else return a.offset + i - l0 + 1;
		}

		template<axis A, domain DM = domain::interior>
		auto vertices() const {
			if constexpr (DM == domain::interior) {
				return flecsi::topo::make_ids<index_space::vertices>(
					get_axis<A>().layout.logical());
			} else if constexpr (DM == domain::extended) {
				const flecsi::topo::narray_impl::axis_layout & l = get_axis<A>().layout;
				return flecsi::topo::make_ids<index_space::vertices>(
					flecsi::util::iota_view(
						l.extended<0>() + is_low<A>(), l.extended<1>() - is_high<A>()));
			}
		}

		template<axis A>
		auto extent() {
			return get_axis<A>().layout.extent();
		}

		double xdelta() { return this->policy_meta().xdelta; }

		double ydelta() { return this->policy_meta().ydelta; }

		double dxdy() { return xdelta() * ydelta(); }

		template<class Sten, class MDColex>
		struct stencil_operator {
			constexpr double &
			operator()(std::size_t i, std::size_t j, Sten dir) {
				return so(i, j)[static_cast<std::ptrdiff_t>(dir)];
			}

			constexpr double
			operator()(std::size_t i, std::size_t j, Sten dir) const {
				return so(i, j)[static_cast<std::ptrdiff_t>(dir)];
			}

			MDColex so;
		};

		template<index_space S, class Sten, class T, flecsi::Privileges P>
		auto stencil_op(
			const flecsi::data::accessor<flecsi::data::dense, T, P> & a) const {
			return stencil_operator<
				Sten,
				std::decay_t<decltype(this->template mdcolex<S>(a))>>{
				this->template mdcolex<S>(a)};
		}

		template<axis A>
		double dx() {
			if constexpr (A == x_axis)
				return xdelta();
			else
				return ydelta();
		}

		template<axis A>
		double value(std::size_t i) {
			return (dx<A>() * global_id<A>(i));
		}

		template<axis A>
		bool is_low() const {
			return get_axis<A>().low();
		}

		template<axis A>
		bool is_high() const {
			return get_axis<A>().high();
		}

		template<index_space Space>
		auto dofs() {
			return util::make_subrange_ids<Space, axes::size>(
				interior_subrange<Space>(axes()), extents_array<Space>(axes()));
		}

		void set_geom(const grect & g) {
			auto & md = this->policy_meta();
			double xdelta =
				std::abs(g[0][1] - g[0][0]) / (get_axis<x_axis>().axis.extent + 1);
			double ydelta =
				std::abs(g[1][1] - g[1][0]) / (get_axis<y_axis>().axis.extent + 1);

			md.xdelta = xdelta;
			md.ydelta = ydelta;
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
				extent<Axis>()...};
		}
	};

	static coloring color(const index_definition & idef) { return {{idef}}; }

	static void set_geometry(mesh::accessor<flecsi::rw> sm, grect const & g) {
		sm.set_geom(g);
	}

	static void initialize(flecsi::scheduler & s,
	                       mesh::topology & m,
	                       coloring const &,
	                       grect const & geometry) {
		flecsi::execute<set_geometry, flecsi::mpi>(m, geometry);
	}
};

static constexpr std::size_t nvecs = 3;
inline std::array<field<double>::definition<mesh, mesh::vertices>, nvecs> ud;

template<class Sten>
using stencil_field =
	field<std::array<double, static_cast<std::size_t>(Sten::ndirs)>>;

inline stencil_field<five_pt>::definition<mesh, mesh::vertices> sod;

}

#endif
