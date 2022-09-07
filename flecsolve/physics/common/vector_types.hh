#pragma once

#include <array>

#include <flecsi/util/array_ref.hh>
#include <flecsi/util/constant.hh>
#include <tuple>
#include <type_traits>
#include <utility>

#include "flecsolve/vectors/data/mesh.hh"

namespace flecsolve {
namespace physics {

template<class Vec>
using scalar_t = typename Vec::scalar;

template<class Vec>
using topo_t = typename Vec::data_t::topo_t;

template<class Vec>
using topo_slot_t = typename Vec::data_t::topo_slot_t;

template<class Vec>
using topo_axes_t = typename topo_t<Vec>::axes;

template<class Vec>
using topo_domain_t = typename topo_t<Vec>::domain;

template<class Vec>
using topo_acc =
	typename topo_t<Vec>::template accessor<flecsi::ro>;
// using topo_acc = typename Vec::data_t::topo_acc;

template<class Vec, flecsi::partition_privilege_t priv>
using field_acc = typename Vec::data_t::template acc<priv>;

template<class Vec, flecsi::partition_privilege_t priv>
using field_acc_all = typename Vec::data_t::template acc_all<priv>;

using vec::data::field;

template<class Vec, auto Space = topo_t<Vec>::space>
using field_def =
	typename field<scalar_t<Vec>>::template definition<topo_t<Vec>, Space>;
template<class Vec, auto Space = topo_t<Vec>::space>
using field_ref =
	typename field<scalar_t<Vec>>::template Reference<topo_t<Vec>, Space>;

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