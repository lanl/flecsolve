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
#ifndef FLECSI_LINALG_VECTORS_TOPO_VIEW_HH
#define FLECSI_LINALG_VECTORS_TOPO_VIEW_HH

#include "core.hh"
#include "variable.hh"
#include "operations/topo_view.hh"
#include "data/topo_view.hh"

namespace flecsolve::vec {

template<auto V, class Scalar, flecsi::data::layout L, class Topo, typename Topo::index_space Space>
struct topo_view_config {
	using scalar = Scalar;
	using real = typename num_traits<scalar>::real;
	using len_t = flecsi::util::id;
	static constexpr auto var = variable<V>;
	using var_t = decltype(V);
	static constexpr std::size_t num_components = 1;
	using topo_t = Topo;
	static constexpr typename Topo::index_space space = Space;
	static constexpr flecsi::data::layout layout = L;
};

template<auto V, class Scalar, flecsi::data::layout L, class Topo, typename Topo::index_space Space>
using topo_view = core<data::topo_view,
                       ops::topo_view,
                       topo_view_config<V, Scalar, L, Topo, Space>>;

template<auto V,
         class T,
         flecsi::data::layout L,
         class Topo,
         typename Topo::index_space Space>
auto make(variable_t<V>,
          flecsi::data::field_reference<T, L, Topo, Space> ref) {
	using Ref = flecsi::data::field_reference<T, L, Topo, Space>;
	using vec_t = topo_view<V, T, L, Topo, Space>;
	using config_t = typename vec_t::config;

	return vec_t{data::topo_view<config_t>{ref}};
}


template<class T,
         flecsi::data::layout L,
         class Topo,
         typename Topo::index_space Space>
auto make(flecsi::data::field_reference<T, L, Topo, Space> ref) {
	return vec::make(variable<anon_var::anonymous>, ref);
}


template<auto V, class T>
auto make(variable_t<V> var, flecsi::topology<T> & topo) {
	return [&, var](auto &... fd) {
		if constexpr (sizeof...(fd) == 1) {
			return make(var, fd(topo)...);
		}
		else {
			return std::tuple(make(var, fd(topo))...);
		}
	};
}

template<class T>
auto make(flecsi::topology<T> & topo) {
	return make(variable<anon_var::anonymous>, topo);
}

}
#endif
