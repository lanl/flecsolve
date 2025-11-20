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
#ifndef FLECSOLVE_VECTORS_OPERATIONS_TOPO_TASKS_HH
#define FLECSOLVE_VECTORS_OPERATIONS_TOPO_TASKS_HH

#include <cmath>
#include <random>
#include <fstream>

#include <flecsi/execution.hh>
#include <flecsi/data.hh>

#include "flecsolve/util/traits.hh"
#include "flecsolve/vectors/data/topo_view.hh"

namespace flecsolve::vec::ops {

using flecsi::ro;
using flecsi::rw;
using flecsi::wo;

template<class VecData, class Scalar, class Len>
struct topo_tasks {

	using scalar = Scalar;
	using real = typename num_traits<Scalar>::real;
	using len = Len;
	static constexpr bool is_complex = num_traits<scalar>::is_complex;
	using topo_acc = typename VecData::topo_acc;

	template<flecsi::privilege priv>
	using acc = typename VecData::template acc<priv>;

	template<flecsi::privilege priv>
	using acc_all = typename VecData::template acc_all<priv>;

	using util = typename VecData::util;

	static scalar scalar_prod(flecsi::exec::accelerator s, topo_acc m, acc<ro> x, acc<ro> y) noexcept {
		scalar res = s.executor().named("scalar_prod")
			.reduceall(dof,
			           up,
			           util::dofs(m),
			           flecsi::exec::fold::sum,
			           scalar) {
			if constexpr (is_complex)
				up(std::conj(y[dof]) * x[dof]);
			else
				up(x[dof] * y[dof]);
		};

		return res;
	}

	static void set_to_scalar(flecsi::exec::accelerator s, topo_acc m, acc<wo> x, scalar val) noexcept {
		s.executor().forall(dof, util::dofs(m)) { x[dof] = val; };
	}

	static void scale_self(flecsi::exec::accelerator s, topo_acc m, acc<rw> x, scalar val) noexcept {
		s.executor().forall(dof, util::dofs(m)) { x[dof] *= val; };
	}

	static void scale(flecsi::exec::accelerator s, topo_acc m, acc<ro> x, acc<wo> y, scalar val) noexcept {
		s.executor().forall(dof, util::dofs(m)) { y[dof] = x[dof] * val; };
	}

	template<class OtherAcc>
	static void copy(flecsi::exec::accelerator s, topo_acc, acc_all<wo> xa, OtherAcc ya) noexcept {
		flecsi::util::iota_view<std::size_t> v(0, xa.span().size());
		s.executor().forall(i, v) { xa[i] = ya[i]; };
	}

	static void add_self(flecsi::exec::accelerator s, topo_acc m, acc<rw> z, acc<ro> x) noexcept {
		s.executor().forall(dof, util::dofs(m)) { z[dof] = z[dof] + x[dof]; };
	}

	static void add(flecsi::exec::accelerator s, topo_acc m, acc<wo> z, acc<ro> x, acc<ro> y) noexcept {
		s.executor().forall(dof, util::dofs(m)) { z[dof] = x[dof] + y[dof]; };
	}

	static void subtract(flecsi::exec::accelerator s, topo_acc m, acc<wo> x, acc<ro> a, acc<ro> b) noexcept {
		s.executor().forall(dof, util::dofs(m)) { x[dof] = a[dof] - b[dof]; };
	}

	template<bool inv>
	static void subtract_self(flecsi::exec::accelerator s, topo_acc m, acc<rw> x, acc<ro> b) noexcept {
		s.executor().forall(dof, util::dofs(m)) {
			if constexpr (inv) {
				x[dof] = x[dof] - b[dof];
			}
			else {
				x[dof] = b[dof] - x[dof];
			}
		};
	}

	static void multiply(flecsi::exec::accelerator s, topo_acc m, acc<wo> z, acc<ro> x, acc<ro> y) noexcept {
		s.executor().forall(dof, util::dofs(m)) { z[dof] = x[dof] * y[dof]; };
	}

	static void multiply_self(flecsi::exec::accelerator s, topo_acc m, acc<rw> x, acc<ro> y) noexcept {
		s.executor().forall(dof, util::dofs(m)) {
			x[dof] = x[dof] * y[dof];
		};
	}

	template<bool inv>
	static void divide_self(flecsi::exec::accelerator s, topo_acc m, acc<rw> z, acc<ro> x) noexcept {
		s.executor().forall(dof, util::dofs(m)) {
			if constexpr (inv) {
				z[dof] = z[dof] / x[dof];
			}
			else {
				z[dof] = x[dof] / z[dof];
			}
		};
	}

	static void divide(flecsi::exec::accelerator s, topo_acc m, acc<wo> z, acc<ro> x, acc<ro> y) noexcept {
		s.executor().forall(dof, util::dofs(m)) { z[dof] = x[dof] / y[dof]; };
	}

	static void reciprocal_self(flecsi::exec::accelerator s, topo_acc m, acc<rw> x) noexcept {
		s.executor().forall(dof, util::dofs(m)) { x[dof] = 1.0 / x[dof]; };
	}

	static void reciprocal(flecsi::exec::accelerator s, topo_acc m, acc<wo> x, acc<ro> y) noexcept {
		s.executor().forall(dof, util::dofs(m)) { x[dof] = 1.0 / y[dof]; };
	}

	static void linear_sum(flecsi::exec::accelerator s,
	                       topo_acc m,
	                       acc<wo> z,
	                       scalar alpha,
	                       acc<ro> x,
	                       scalar beta,
	                       acc<ro> y) noexcept {

		s.executor().forall(dof, util::dofs(m)) {
			z[dof] = alpha * x[dof] + beta * y[dof];
		};
	}

	template<bool inv>
	static void linear_sum_self(flecsi::exec::accelerator s,
	                            topo_acc m,
	                            acc<rw> z,
	                            acc<ro> x,
	                            scalar alpha,
	                            scalar beta) noexcept {
		s.executor().forall(dof, util::dofs(m)) {
			if constexpr (inv) {
				z[dof] = alpha * z[dof] + beta * x[dof];
			}
			else {
				z[dof] = alpha * x[dof] + beta * z[dof];
			}
		};
	}

	static void
	axpy(flecsi::exec::accelerator s, topo_acc m, acc<wo> z, scalar alpha, acc<ro> x, acc<ro> y) noexcept {
		s.executor().forall(dof, util::dofs(m)) {
			z[dof] = alpha * x[dof] + y[dof];
		};
	}

	template<bool inv>
	static void axpy_self(flecsi::exec::accelerator s, topo_acc m, acc<rw> z, acc<ro> x, scalar alpha) noexcept {
		s.executor().forall(dof, util::dofs(m)) {
			if constexpr (inv) {
				z[dof] = alpha * z[dof] + x[dof];
			}
			else {
				z[dof] = alpha * x[dof] + z[dof];
			}
		};
	}

	static void
	axpby(flecsi::exec::accelerator s, topo_acc m, acc<rw> y, acc<ro> x, scalar alpha, scalar beta) noexcept {
		s.executor().forall(dof, util::dofs(m)) {
			y[dof] = alpha * x[dof] + beta * y[dof];
		};
	}

	static void abs_self(flecsi::exec::accelerator s, topo_acc m, acc<rw> x) noexcept {
		s.executor().forall(dof, util::dofs(m)) { x[dof] = std::abs(x[dof]); };
	}

	static void abs(flecsi::exec::accelerator s, topo_acc m, acc<wo> y, acc<ro> x) noexcept {
		s.executor().forall(dof, util::dofs(m)) { y[dof] = std::abs(x[dof]); };
	}

	static void add_scalar_self(flecsi::exec::accelerator s, topo_acc m, acc<rw> x, scalar alpha) noexcept {
		s.executor().forall(dof, util::dofs(m)) { x[dof] += alpha; };
	}

	static void add_scalar(flecsi::exec::accelerator s, topo_acc m, acc<wo> y, acc<ro> x, scalar alpha) noexcept {
		s.executor().forall(dof, util::dofs(m)) { y[dof] = x[dof] + alpha; };
	}

	static real lp_norm_local(flecsi::exec::accelerator s, topo_acc m, acc<ro> u, int p) noexcept {
		real ret = s.executor().reduceall(dof,
		                                  up,
		                                  util::dofs(m),
		                                  flecsi::exec::fold::sum,
		                                  real) {
			up(std::pow(u[dof], p));
		};

		return ret;
	}

	static real l1_norm_local(flecsi::exec::accelerator s, topo_acc m, acc<ro> u) noexcept {
		real ret = s.executor().reduceall(
			dof, up, util::dofs(m), flecsi::exec::fold::sum, real) {
			up(std::abs(u[dof]));
		};

		return ret;
	}

	static real l2_norm_local(flecsi::exec::accelerator s, topo_acc m, acc<ro> u) noexcept  {
		auto ret = scalar_prod(s, m, u, u);
		if constexpr (is_complex)
			return ret.real();
		else
			return ret;
	}

	static real local_max(flecsi::exec::accelerator s, topo_acc m, acc<ro> u) noexcept {
		if constexpr (is_complex) {
			auto ret = s.executor().reduceall(dof,
			                                  up,
			                                  util::dofs(m),
			                                  flecsi::exec::fold::max,
			                                  real) {
				up(u[dof].real());
			};
			return ret;
		}
		else {
			auto ret = s.executor().reduceall(dof,
			                                  up,
			                                  util::dofs(m),
			                                  flecsi::exec::fold::max,
			                                  real) {
				up(u[dof]);
			};
			return ret;
		}
	}

	static real local_min(flecsi::exec::accelerator s, topo_acc m, acc<ro> u) noexcept {
		if constexpr (is_complex) {
			auto ret = s.executor().reduceall(dof,
			                                  up,
			                                  util::dofs(m),
			                                  flecsi::exec::fold::min,
			                                  real) {
				up(u[dof].real());
			};
			return ret;
		}
		else {
			auto ret = s.executor().reduceall(dof,
			                                  up,
			                                  util::dofs(m),
			                                  flecsi::exec::fold::min,
			                                  real) {
				up(u[dof]);
			};
			return ret;
		}
	}

	static real inf_norm_local(flecsi::exec::accelerator s, topo_acc m, acc<ro> x) noexcept {
		auto ret = s.executor().reduceall(
			dof, up, util::dofs(m), flecsi::exec::fold::max, real) {
			up(std::abs(x[dof]));
		};
		return ret;
	}

	static len local_size(topo_acc m) noexcept { return util::dofs(m).size(); }

	static void get_local_size(topo_acc m, len * length) noexcept {
		*length = util::dofs(m).size();
	}

	static void set_random(topo_acc m, acc<wo> x, unsigned seed) noexcept {
		std::mt19937 gen(seed);
		std::uniform_real_distribution<real> dis(0., 1.);
		for (auto dof : util::dofs(m)) {
			if constexpr (is_complex)
				x[dof] = scalar(dis(gen), dis(gen));
			else
				x[dof] = dis(gen);
		};
	}

	static void dump(flecsi::exec::accelerator s, std::string_view pre, topo_acc m, acc<ro> x) {
		std::string fname{pre};
		fname += "-" + std::to_string(s.launch().index);
		std::ofstream ofile(fname);
		for (auto dof : util::dofs(m)) {
			ofile << x(dof) << '\n';
		}
	}
};

}
#endif
