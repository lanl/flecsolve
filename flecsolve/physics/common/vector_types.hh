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
#pragma once

#include <array>

#include <flecsi/util/array_ref.hh>
#include <flecsi/util/constant.hh>
#include <tuple>
#include <type_traits>
#include <utility>

#include "flecsolve/vectors/data/topo_view.hh"

namespace flecsolve {
namespace physics {

template<class Vec>
using scalar_t = typename Vec::scalar;

template<class Vec>
using topo_t = typename Vec::data_t::topo_t;

template<class Vec>
using topo_axes_t = typename topo_t<Vec>::axes;

template<class Vec>
using topo_domain_t = typename topo_t<Vec>::domain;

template<class Vec>
using topo_acc = typename topo_t<Vec>::template accessor<flecsi::ro>;
// using topo_acc = typename Vec::data_t::topo_acc;

template<class Vec, flecsi::privilege priv>
using field_acc = typename Vec::data_t::template acc<priv>;

template<class Vec, flecsi::privilege priv>
using field_acc_all = typename Vec::data_t::template acc_all<priv>;

template<class Vec, auto Space = topo_t<Vec>::space>
using field_def =
	typename Vec::data_t::template field<scalar_t<Vec>>::template definition<topo_t<Vec>, Space>;
template<class Vec, auto Space = topo_t<Vec>::space>
using field_ref =
	typename Vec::data_t::template field<scalar_t<Vec>>::template Reference<topo_t<Vec>, Space>;

template<class Vec>
using face_def = field_def<Vec, topo_t<Vec>::faces>;

template<class Vec>
using face_ref = field_ref<Vec, topo_t<Vec>::faces>;

template<class Vec>
using cell_def = field_def<Vec, topo_t<Vec>::cells>;

template<class Vec>
using cell_ref = field_ref<Vec, topo_t<Vec>::cells>;

template<class T, class Vec>
using axes_set = flecsi::util::key_array<T, topo_axes_t<Vec>>;

namespace components {

template<class Vec>
using faces_handle_single = std::optional<face_ref<Vec>>;

template<class Vec>
using faces_handle = std::optional<axes_set<face_ref<Vec>, Vec>>;

template<class Vec>
using cells_handle = std::optional<cell_ref<Vec>>;

}

}
}
