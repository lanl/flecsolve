#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <flecsi/util/array_ref.hh>
#include <flecsi/util/constant.hh>
#include <tuple>
#include <utility>

#include "flecsi-linalg/discrete_operators/common/operator_base.hh"
#include "flecsi-linalg/discrete_operators/common/operator_utils.hh"
#include "flecsi-linalg/discrete_operators/common/state_store.hh"
#include "flecsi-linalg/discrete_operators/tasks/operator_task.hh"

namespace flecsi {
namespace linalg {
namespace discrete_operators {

/**
 * @brief operator of finite-volume diffusion
 *
 * linear operators of the form -β ∇ (b ∇ u ) + α a u
 * where α and β  are constants, a is a cell-centered
 * array,b is a face-centered array, and u is a cell-centered
 * array.
 *
 * @tparam Var variable to apply on
 * @tparam Topo topology
 * @tparam Scalar scalar data-type
 */
template <auto Var, class Topo, class Scalar = double>
struct volume_diffusion_op;

template <auto Var, class Topo, class Scalar>
struct operator_traits<volume_diffusion_op<Var, Topo, Scalar>> {
  using scalar_t = Scalar;
  using topo_t = Topo;
  using topo_slot_t = flecsi::data::topology_slot<Topo>;
  using topo_axes_t = typename topo_t::axes;
  constexpr static auto dim = Topo::dimension;
  using tasks_f = tasks::topology_tasks<topo_t, field<scalar_t>>;

  using cell_def =
      typename field<scalar_t>::template definition<topo_t, topo_t::cells>;
  using cell_ref =
      typename field<scalar_t>::template Reference<topo_t, topo_t::cells>;

  using face_def =
      typename field<scalar_t>::template definition<topo_t, topo_t::faces>;
  using face_ref =
      typename field<scalar_t>::template Reference<topo_t, topo_t::faces>;
};

template <auto Var, class Topo, class Scalar>
struct operator_parameters<volume_diffusion_op<Var, Topo, Scalar>> {
  using op_type = volume_diffusion_op<Var, Topo, Scalar>;
  using cell_ref = typename operator_traits<op_type>::cell_ref;
  using face_ref = typename operator_traits<op_type>::face_ref;
  Scalar beta;
  Scalar alpha;

  std::optional<cell_ref> a;
  std::optional<face_ref> b;
};

template <auto Var, class Topo, class Scalar>
struct volume_diffusion_op
    : operator_settings<volume_diffusion_op<Var, Topo, Scalar>> {

  using base_type = operator_settings<volume_diffusion_op<Var, Topo, Scalar>>;
  using exact_type = typename base_type::exact_type;
  using param_type = typename base_type::param_type;

  using scalar_t = typename operator_traits<exact_type>::scalar_t;
  using topo_t = typename operator_traits<exact_type>::topo_t;

  // using diffop_t = DiffOp<my_t>;
  using topo_slot_t = typename operator_traits<exact_type>::topo_slot_t;
  using topo_axes_t = typename operator_traits<exact_type>::topo_axes_t;
  using cell_def = typename operator_traits<exact_type>::cell_def;
  using cell_ref = typename operator_traits<exact_type>::cell_ref;
  using face_def = typename operator_traits<exact_type>::face_def;
  using face_ref = typename operator_traits<exact_type>::face_ref;

  using tasks_f = typename operator_traits<exact_type>::tasks_f;

  static constexpr std::size_t dim = operator_traits<exact_type>::dim;

  using flux_store_t =
      common::topo_state_store<exact_type, topo_t::faces, dim, 0>;
  // zero-D quantity change due to flux integration on surface
  using du_store_t = common::topo_state_store<exact_type, topo_t::cells, 1, 0>;

  // These arrays hold references to the fields provided by the above
  // declarations
  std::array<face_ref, dim> fluxes;

  cell_ref du;

  volume_diffusion_op(topo_slot_t &s, param_type parameters)
      : base_type(parameters), fluxes(flux_store_t::get_state(s)),
        du(du_store_t::get_state(s)) {}

  template <class U, class V> constexpr auto apply(U &&u, V &&v) const {
    auto subu = u.template getvar<Var>();
    auto subv = v.template getvar<Var>();
    _apply(subu.data.topo, subu.data.ref(), subv.data.ref());
    //_apply(u.data.topo, u.data.ref(), v.data.ref());
  }

  void _apply(topo_slot_t &m, cell_ref u, cell_ref v) const {
    // first, zero-out the fields to take results
    execute<tasks_f::zero_op>(m, v);
    execute<tasks_f::zero_op>(m, du);

    // determine the fluxes along the axis
    sweep(m, u, topo_axes_t());

    // collect all prior calculations and apply to range vector
    execute<tasks_f::diffuse_op>(m, this->parameters.beta,
                                 this->parameters.alpha, *(this->parameters.a),
                                 u, du, v);
  }

  template <auto... Axis>
  constexpr void sweep(topo_slot_t &m, cell_ref u,
                       util::constants<Axis...>) const {
    (execute<tasks_f::template flux_op<Axis>>(m, u, *(this->parameters.b),
                                              fluxes[Axis]),
     ...);
    (execute<tasks_f::template flux_sum<Axis>>(m, fluxes[Axis], du), ...);
  }
};

} // namespace discrete_operators
} // namespace linalg
} // namespace flecsi
