#pragma once

#include <cmath>

#include <flecsi/data.hh>
#include <flecsi/execution.hh>

#include "flecsi-linalg/vectors/data/flecsi_data.hh"


namespace flecsi::linalg {

static constexpr flecsi::partition_privilege_t na = flecsi::na;
static constexpr flecsi::partition_privilege_t ro = flecsi::ro;
static constexpr flecsi::partition_privilege_t wo = flecsi::wo;
static constexpr flecsi::partition_privilege_t rw = flecsi::rw;

template<typename T, flecsi::data::layout L = flecsi::data::layout::dense>
using field = flecsi::field<T, L>;

template<class topo, typename topo::index_space space, class real = double>
struct flecsi_tasks {
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


	static void set_to_scalar(topo_acc m,
	                          wo_acc x,
	                          real val) {
		for (auto dof : m.template dofs<space>()) {
			x[dof] = val;
		}
	}


	static void scale_self(topo_acc m,
	                       rw_acc x,
	                       real val) {
		for (auto dof : m.template dofs<space>()) {
			x[dof] *= val;
		}
	}

	static void scale(topo_acc m,
	                  ro_acc x,
	                  wo_acc y,
	                  real val) {
		for (auto dof : m.template dofs<space>()) {
			y[dof] = x[dof] * val;
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

	static void add_self(topo_acc m,
	                     wo_acc z,
	                     ro_acc x) {
		for (auto dof : m.template dofs<space>()) {
			z[dof] = z[dof] + x[dof];
		}
	}

	static void add(topo_acc m,
	                wo_acc z,
	                ro_acc x,
	                ro_acc y) {
		for (auto dof : m.template dofs<space>()) {
			z[dof] = x[dof] + y[dof];
		}
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

	template<bool inv>
	static void subtract_self(topo_acc m,
	                          wo_acc x,
	                          ro_acc b)
	{
		for (auto dof : m.template dofs<space>()) {
			if constexpr (inv) {
				x[dof] = x[dof] - b[dof];
			} else {
				x[dof] = b[dof] - x[dof];
			}
		}
	}

	static void multiply(topo_acc m,
	                     wo_acc z,
	                     ro_acc x,
	                     ro_acc y) {
		for (auto dof : m.template dofs<space>()) {
			z[dof] = x[dof] * y[dof];
		}
	}


	static void multiply_self(topo_acc m,
	                          rw_acc x,
	                          ro_acc y) {
		for (auto dof : m.template dofs<space>()) {
			x[dof] = x[dof] * y[dof];
		}
	}

	template<bool inv>
	static void divide_self(topo_acc m,
	                        rw_acc z,
	                        ro_acc x) {
		for (auto dof : m.template dofs<space>()) {
			if constexpr (inv) {
				z[dof] = z[dof] / x[dof];
			} else {
				z[dof] = x[dof] / z[dof];
			}
		}
	}

	static void divide(topo_acc m,
	                   wo_acc z,
	                   ro_acc x,
	                   ro_acc y) {
		for (auto dof : m.template dofs<space>()) {
			z[dof] = x[dof] / y[dof];
		}
	}

	static void reciprocal_self(topo_acc m,
	                            rw_acc x) {
		for (auto dof : m.template dofs<space>()) {
			x[dof] = 1.0 / x[dof];
		}
	}

	static void reciprocal(topo_acc m,
	                       wo_acc x,
	                       ro_acc y) {
		for (auto dof : m.template dofs<space>()) {
			x[dof] = 1.0 / y[dof];
		}
	}

	static void linear_sum(topo_acc m,
	                       wo_acc z,
	                       real alpha,
	                       ro_acc x,
	                       real beta,
	                       ro_acc y)
	{

		for (auto dof : m.template dofs<space>()) {
			z[dof] = alpha * x[dof] + beta * y[dof];
		}
	}

	template<bool inv>
	static void linear_sum_self(topo_acc m,
	                            rw_acc z,
	                            ro_acc x, real alpha, real beta)
	{
		for (auto dof : m.template dofs<space>()) {
			if constexpr (inv) {
				z[dof] = alpha * z[dof] + beta * x[dof];
			} else {
				z[dof] = alpha * x[dof] + beta * z[dof];
			}
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

	template<bool inv>
	static void axpy_self(topo_acc m,
	                      rw_acc z,
	                      ro_acc x,
	                      real alpha) {
		for (auto dof : m.template dofs<space>()) {
			if constexpr (inv) {
				z[dof] = alpha * z[dof] + x[dof];
			} else {
				z[dof] = alpha * x[dof] + z[dof];
			}
		}
	}

	static void axpby(topo_acc m,
	                  rw_acc y,
	                  ro_acc x,
	                  real alpha,
	                  real beta) {
		for (auto dof : m.template dofs<space>()) {
			y[dof] = alpha * x[dof] + beta * y[dof];
		}
	}

	static void abs_self(topo_acc m,
	                     rw_acc x) {
		for (auto dof : m.template dofs<space>()) {
			x[dof] = std::abs(x[dof]);
		}
	}

	static void abs(topo_acc m,
	                wo_acc y,
	                ro_acc x) {
		for (auto dof : m.template dofs<space>()) {
			y[dof] = std::abs(x[dof]);
		}
	}

	static void add_scalar_self(topo_acc m,
	                            rw_acc x,
	                            real alpha) {
		for (auto dof : m.template dofs<space>()) {
			x[dof] += alpha;
		}
	}

	static void add_scalar(topo_acc m,
	                       wo_acc y,
	                       ro_acc x,
	                       real alpha) {
		for (auto dof : m.template dofs<space>()) {
			y[dof] = x[dof] + alpha;
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

	static real l1_norm_local(topo_acc m,
	                          ro_acc u) {
		real ret = 0;
		for (auto dof : m.template dofs<space>()) {
			ret += std::abs(u[dof]);
		}

		return ret;
	}

	static real l2_norm_local(topo_acc m,
	                          ro_acc u) {
		real ret = 0;
		for (auto dof : m.template dofs<space>()) {
			ret += u[dof] * u[dof];
		}

		return ret;
	}

	static real local_max(topo_acc m,
	                      ro_acc u) {
		auto ret = std::numeric_limits<real>::min();
		for (auto dof : m.template dofs<space>()) {
			ret = std::max(u[dof], ret);
		}
		return ret;
	}

	static real local_min(topo_acc m,
	                      ro_acc u) {
		auto ret = std::numeric_limits<real>::max();
		for (auto dof : m.template dofs<space>()) {
			ret = std::min(u[dof], ret);
		}
		return ret;
	}

	static real inf_norm_local(topo_acc m,
	                           ro_acc x) {
		auto ret = std::numeric_limits<real>::min();
		for (auto dof : m.template dofs<space>()) {
			ret = std::max(std::abs(x[dof]), ret);
		}
		return ret;
	}
};

}
