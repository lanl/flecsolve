#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <utility>
#include <vector>

#include "operators/operator_task.hh"
#include "operators/specialization/operator_mesh.hh"

namespace flecsi
namespace linalg
namespace discrete_operators {

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar = double,
         template<class> class MixinSet = make_mixins<>::templ>
struct neumann;

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar,
         template<class>
         class MixinSet>
struct operator_traits<neumann<Var, Topo, Axis, Boundary, Scalar, MixinSet>>
{
  using scalar_t = Scalar;
  using topo_t = Topo;
  using topo_slot_t = flecsi::data::topology_slot<Topo>;
  using topo_axes_t = typename topo_t::axes;
  constexpr static auto dim = Topo::dimension;
  using tasks_f =
      linalg::operators::tasks::topology_tasks<topo_t, field<scalar_t>>;

  using cell_ref =
      typename field<scalar_t>::template Reference<topo_t, topo_t::cells>;

  using face_ref =
      typename field<scalar_t>::template Reference<topo_t, topo_t::faces>;

  constexpr static auto op_axis = Axis;
  constexpr static auto op_boundary = Boundary;

  struct Params
  {
    // const face_ref b;
    std::optional<face_ref> b;
  };
};

template<auto Var,
         class Topo,
         typename Topo::axis Axis,
         typename Topo::domain Boundary,
         class Scalar,
         template<class>
         class MixinSet>
struct neumann
    : operator_host<neumann<Var, Topo, Axis, Boundary, Scalar, MixinSet>,
                    MixinSet>
{
  using base_type =
      operator_host<neumann<Var, Topo, Axis, Boundary, Scalar, MixinSet>,
                    MixinSet>;
  using exact_type = typename base_type::exact_type;
  using param_type = typename base_type::param_type;

  using topo_slot_t = typename operator_traits<exact_type>::topo_slot_t;
  using cell_ref = typename operator_traits<exact_type>::cell_ref;
  using tasks_f = typename operator_traits<exact_type>::tasks_f;

  neumann(param_type p) : base_type(p) {}

  template<class U, class V>
  constexpr auto apply(U&& u, V&& v) const
  {
    // if constexpr (is_tuple<typename std::decay_t<U>::data_t>::value) {
    auto subu = u.template getvar<Var>();
    _apply(subu.data.topo, subu.data.ref());
  }

  void _apply(topo_slot_t& m, cell_ref u) const
  {
    flecsi::execute<tasks_f::template boundary_fluxset<Axis, Boundary>>(
        m, *(this->params.b), u);
  }
};

}
}
}  // namespace flecsi::linalg::operators
