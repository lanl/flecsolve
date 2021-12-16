#pragma once

#include <cmath>

#include <flecsi/execution.hh>

#include "flecsi-linalg/vectors/data/flecsi_data.hh"


namespace flecsi::linalg {

template<class VecData>
struct flecsi_tasks {

	using real = typename VecData::real_t;
	using len = typename VecData::len_t;
	using topo_acc = typename VecData::topo_acc;
	using ro_acc = typename VecData::ro_acc;
	using wo_acc = typename VecData::wo_acc;
	using rw_acc = typename VecData::rw_acc;

	using ro_acc_all = typename VecData::ro_acc_all;
	using wo_acc_all = typename VecData::wo_acc_all;

	using util = typename VecData::util;

	static real prod(topo_acc m,
	                 ro_acc x,
	                 ro_acc y) {
		real res = 0.0;

		for (auto dof : util::dofs(m)) {
			res += x[dof] * y[dof];
		}

		return res;
	}


	static void set_to_scalar(topo_acc m,
	                          wo_acc x,
	                          real val) {
		for (auto dof : util::dofs(m)) {
			x[dof] = val;
		}
	}


	static void scale_self(topo_acc m,
	                       rw_acc x,
	                       real val) {
		for (auto dof : util::dofs(m)) {
			x[dof] *= val;
		}
	}

	static void scale(topo_acc m,
	                  ro_acc x,
	                  wo_acc y,
	                  real val) {
		for (auto dof : util::dofs(m)) {
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
		for (auto dof : util::dofs(m)) {
			z[dof] = z[dof] + x[dof];
		}
	}

	static void add(topo_acc m,
	                wo_acc z,
	                ro_acc x,
	                ro_acc y) {
		for (auto dof : util::dofs(m)) {
			z[dof] = x[dof] + y[dof];
		}
	}

	static void subtract(topo_acc m,
	                     wo_acc x,
	                     ro_acc a,
	                     ro_acc b)
	{
		for (auto dof : util::dofs(m)) {
			x[dof] = a[dof] - b[dof];
		}
	}

	template<bool inv>
	static void subtract_self(topo_acc m,
	                          wo_acc x,
	                          ro_acc b)
	{
		for (auto dof : util::dofs(m)) {
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
		for (auto dof : util::dofs(m)) {
			z[dof] = x[dof] * y[dof];
		}
	}


	static void multiply_self(topo_acc m,
	                          rw_acc x,
	                          ro_acc y) {
		for (auto dof : util::dofs(m)) {
			x[dof] = x[dof] * y[dof];
		}
	}

	template<bool inv>
	static void divide_self(topo_acc m,
	                        rw_acc z,
	                        ro_acc x) {
		for (auto dof : util::dofs(m)) {
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
		for (auto dof : util::dofs(m)) {
			z[dof] = x[dof] / y[dof];
		}
	}

	static void reciprocal_self(topo_acc m,
	                            rw_acc x) {
		for (auto dof : util::dofs(m)) {
			x[dof] = 1.0 / x[dof];
		}
	}

	static void reciprocal(topo_acc m,
	                       wo_acc x,
	                       ro_acc y) {
		for (auto dof : util::dofs(m)) {
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

		for (auto dof : util::dofs(m)) {
			z[dof] = alpha * x[dof] + beta * y[dof];
		}
	}

	template<bool inv>
	static void linear_sum_self(topo_acc m,
	                            rw_acc z,
	                            ro_acc x, real alpha, real beta)
	{
		for (auto dof : util::dofs(m)) {
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
		for (auto dof : util::dofs(m)) {
			z[dof] = alpha * x[dof] + y[dof];
		}
	}

	template<bool inv>
	static void axpy_self(topo_acc m,
	                      rw_acc z,
	                      ro_acc x,
	                      real alpha) {
		for (auto dof : util::dofs(m)) {
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
		for (auto dof : util::dofs(m)) {
			y[dof] = alpha * x[dof] + beta * y[dof];
		}
	}

	static void abs_self(topo_acc m,
	                     rw_acc x) {
		for (auto dof : util::dofs(m)) {
			x[dof] = std::abs(x[dof]);
		}
	}

	static void abs(topo_acc m,
	                wo_acc y,
	                ro_acc x) {
		for (auto dof : util::dofs(m)) {
			y[dof] = std::abs(x[dof]);
		}
	}

	static void add_scalar_self(topo_acc m,
	                            rw_acc x,
	                            real alpha) {
		for (auto dof : util::dofs(m)) {
			x[dof] += alpha;
		}
	}

	static void add_scalar(topo_acc m,
	                       wo_acc y,
	                       ro_acc x,
	                       real alpha) {
		for (auto dof : util::dofs(m)) {
			y[dof] = x[dof] + alpha;
		}
	}

	static real lp_norm_local(topo_acc m,
	                          ro_acc u,
	                          int p) {
		real ret = 0;
		for (auto dof : util::dofs(m)) {
			ret += std::pow(u[dof], p);
		}

		return ret;
	}

	static real l1_norm_local(topo_acc m,
	                          ro_acc u) {
		real ret = 0;
		for (auto dof : util::dofs(m)) {
			ret += std::abs(u[dof]);
		}

		return ret;
	}

	static real l2_norm_local(topo_acc m,
	                          ro_acc u) {
		real ret = 0;
		for (auto dof : util::dofs(m)) {
			ret += u[dof] * u[dof];
		}

		return ret;
	}

	static real local_max(topo_acc m,
	                      ro_acc u) {
		auto ret = std::numeric_limits<real>::min();
		for (auto dof : util::dofs(m)) {
			ret = std::max(u[dof], ret);
		}
		return ret;
	}

	static real local_min(topo_acc m,
	                      ro_acc u) {
		auto ret = std::numeric_limits<real>::max();
		for (auto dof : util::dofs(m)) {
			ret = std::min(u[dof], ret);
		}
		return ret;
	}

	static real inf_norm_local(topo_acc m,
	                           ro_acc x) {
		auto ret = std::numeric_limits<real>::min();
		for (auto dof : util::dofs(m)) {
			ret = std::max(std::abs(x[dof]), ret);
		}
		return ret;
	}

	static len local_size(topo_acc m) {
		return util::dofs(m).size();
	}
};

}
