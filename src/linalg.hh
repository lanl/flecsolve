#pragma once

#include <cmath>

#include <flecsi/data.hh>
#include <flecsi/execution.hh>

namespace linalg {

static constexpr flecsi::partition_privilege_t na = flecsi::na;
static constexpr flecsi::partition_privilege_t ro = flecsi::ro;
static constexpr flecsi::partition_privilege_t wo = flecsi::wo;
static constexpr flecsi::partition_privilege_t rw = flecsi::rw;

template<typename T, flecsi::data::layout L = flecsi::data::layout::dense>
using field = flecsi::field<T, L>;

template<class topo, typename topo::index_space space, class real = double>
struct tasks {
  static inline constexpr flecsi::PrivilegeCount num_priv =
    topo::template privilege_count<space>;
  static inline constexpr flecsi::Privileges read_only_dofs =
    flecsi::privilege_cat<flecsi::privilege_repeat<ro, num_priv - (num_priv > 1)>,
                          flecsi::privilege_repeat<na, (num_priv > 1)>>;
  static inline constexpr flecsi::Privileges write_only_dofs =
    flecsi::privilege_cat<flecsi::privilege_repeat<wo, num_priv - (num_priv > 1)>,
                          flecsi::privilege_repeat<na, (num_priv > 1)>>;
  static inline constexpr flecsi::Privileges read_write_dofs =
    flecsi::privilege_cat<flecsi::privilege_repeat<rw, num_priv - (num_priv > 1)>,
                          flecsi::privilege_repeat<na, (num_priv > 1)>>;
  static inline constexpr flecsi::Privileges read_only_all =
    flecsi::privilege_repeat<ro, num_priv>;
  static inline constexpr flecsi::Privileges write_only_all =
    flecsi::privilege_repeat<wo, num_priv>;

  using topo_acc = typename topo::template accessor<ro>;
  using ro_acc = typename field<real>::template accessor1<read_only_dofs>;
  using wo_acc = typename field<real>::template accessor1<write_only_dofs>;
  using rw_acc = typename field<real>::template accessor1<read_write_dofs>;

  using ro_acc_all = typename field<real>::template accessor1<read_only_all>;
  using wo_acc_all = typename field<real>::template accessor1<write_only_all>;


  static real prod(topo_acc m,
                   ro_acc x,
                   ro_acc y) {
    real res = 0.0;

    for (auto dof : m.template dofs<space>()) {
      res += x[dof] * y[dof];
    }

    return res;
  }


  static void set_scalar(topo_acc m,
                         wo_acc x,
                         real val) {
    for (auto dof : m.template dofs<space>()) {
      x[dof] = val;
    }
  }


  static void scale_field(topo_acc m,
                          rw_acc x,
                          real val) {
    for (auto dof : m.template dofs<space>()) {
      x[dof] *= val;
    }
  }


  static void copy(topo_acc m,
                   wo_acc_all xa,
                   ro_acc_all ya) {
    const auto in = ya.span();
    auto out = xa.span();
    std::copy(in.begin(), in.end(),
              out.begin());
  }


  static void subtract(topo_acc m,
                       wo_acc x,
                       ro_acc a,
                       ro_acc b)
  {
    for (auto dof : m.template dofs<space>()) {
      x[dof] = a[dof] - b[dof];
    }
  }


  static void linear_sum(topo_acc m,
                         real alpha,
                         rw_acc x,
                         real beta,
                         ro_acc y)
  {

    for (auto dof : m.template dofs<space>()) {
      x[dof] = alpha * x[dof] + beta * y[dof];
    }
  }


  static void axpy(topo_acc m,
                   wo_acc z,
                   real alpha,
                   ro_acc x,
                   ro_acc y) {
    for (auto dof : m.template dofs<space>()) {
      z[dof] = alpha * x[dof] + y[dof];
    }
  }


  static real lp_norm_local(topo_acc m,
                            ro_acc u,
                            int p) {
    real ret = 0;
    for (auto dof : m.template dofs<space>()) {
      ret += std::pow(u[dof], p);
    }

    return ret;
  }
};



template<class Topo, typename Topo::index_space Space, class Real=double>
struct vec
{
  using real_t = Real;
  using field_definition = typename flecsi::field<real_t>::template definition<Topo, Space>;
  using task = tasks<Topo, Space, Real>;
  using vec_t = vec<Topo, Space, Real>;

  void copy(const vec_t & other) {
    flecsi::execute<task::copy>(topo, def(topo), other.def(other.topo));
  }

  real_t inner_prod(const vec_t & y) const {
    return flecsi::reduce<task::prod,
                          flecsi::exec::fold::sum>(topo, def(topo), y.def(y.topo)).get();
  }

  template<int p>
  real_t lp_norm() const {
    auto val = flecsi::reduce<task::lp_norm_local, flecsi::exec::fold::sum>(topo, def(topo), p).get();
    return std::pow(val, 1./p);
  }

  void subtract(const vec_t & a, const vec_t & b) {
    if (a.def(topo).fid() == def(topo).fid())
      flecsi::execute<task::linear_sum>(topo, 1.0, def(topo), -1, b.def(b.topo));
    else if (b.def(topo).fid() == def(topo).fid())
      flecsi::execute<task::linear_sum>(topo, -1.0, def(topo), 1.0, a.def(a.topo));
    else
      flecsi::execute<task::subtract>(topo, def(topo), a.def(a.topo), b.def(b.topo));
  }

  void scale(real_t val) {
    flecsi::execute<task::scale_field>(topo, def(topo), val);
  }

  void set_scalar(real_t val) {
    flecsi::execute<task::set_scalar>(topo, def(topo), val);
  }

  void axpy(real_t alpha, const vec_t & x, const vec_t & y) {
    if (def(topo).fid() == x.def(x.topo).fid())
      flecsi::execute<task::linear_sum>(topo, alpha, def(topo), 1, y.def(y.topo));
    else if (def(topo).fid() == y.def(y.topo).fid())
      flecsi::execute<task::linear_sum>(topo, 1, def(topo), alpha, x.def(x.topo));
    else
      flecsi::execute<task::axpy>(topo, def(topo), alpha, x.def(x.topo), y.def(y.topo));
  }

  const field_definition & def;
  flecsi::data::topology_slot<Topo> & topo;
};

template<class vec>
struct op
{
  using field_definition = typename vec::field_definition;
  using real_t = typename vec::real_t;

  void apply(const vec & x, vec & y) const {
    apply_fun(x, y);
  }

  void residual(const vec & b, const vec & x,
                vec & r) const {
    apply(x, r);
    r.subtract(b, r);
  }

  std::function<void(const vec & x, vec & y)> apply_fun;
};


}
