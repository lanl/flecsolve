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

	template<flecsi::partition_privilege_t priv>
	using acc = typename VecData::template acc<priv>;

	template<flecsi::partition_privilege_t priv>
	using acc_all = typename VecData::template acc_all<priv>;

	using util = typename VecData::util;

	static scalar scalar_prod(topo_acc m, acc<ro> x, acc<ro> y) {
		scalar res = reduceall(dof,
		                       up,
		                       util::dofs(m),
		                       flecsi::exec::fold::sum,
		                       scalar,
		                       "scalar_prod") {
			if constexpr (is_complex)
				up(std::conj(y[dof]) * x[dof]);
			else
				up(x[dof] * y[dof]);
		};

		return res;
	}

	static void set_to_scalar(topo_acc m, acc<wo> x, scalar val) {
		forall(dof, util::dofs(m), "set_scalar") { x[dof] = val; };
	}

	static void scale_self(topo_acc m, acc<rw> x, scalar val) {
		forall(dof, util::dofs(m), "scale_self") { x[dof] *= val; };
	}

	static void scale(topo_acc m, acc<ro> x, acc<wo> y, scalar val) {
		forall(dof, util::dofs(m), "scale") { y[dof] = x[dof] * val; };
	}

	template<class OtherAcc>
	static void copy(topo_acc, acc_all<wo> xa, OtherAcc ya) {
		flecsi::util::iota_view<std::size_t> v(0, xa.span().size());
		forall(i, v, "copy") { xa[i] = ya[i]; };
	}

	static void add_self(topo_acc m, acc<wo> z, acc<ro> x) {
		forall(dof, util::dofs(m), "add_self") { z[dof] = z[dof] + x[dof]; };
	}

	static void add(topo_acc m, acc<wo> z, acc<ro> x, acc<ro> y) {
		forall(dof, util::dofs(m), "add") { z[dof] = x[dof] + y[dof]; };
	}

	static void subtract(topo_acc m, acc<wo> x, acc<ro> a, acc<ro> b) {
		forall(dof, util::dofs(m), "subtract") { x[dof] = a[dof] - b[dof]; };
	}

	template<bool inv>
	static void subtract_self(topo_acc m, acc<wo> x, acc<ro> b) {
		forall(dof, util::dofs(m), "subtract_self") {
			if constexpr (inv) {
				x[dof] = x[dof] - b[dof];
			}
			else {
				x[dof] = b[dof] - x[dof];
			}
		};
	}

	static void multiply(topo_acc m, acc<wo> z, acc<ro> x, acc<ro> y) {
		forall(dof, util::dofs(m), "multiply") { z[dof] = x[dof] * y[dof]; };
	}

	static void multiply_self(topo_acc m, acc<rw> x, acc<ro> y) {
		forall(dof, util::dofs(m), "multiply_self") {
			x[dof] = x[dof] * y[dof];
		};
	}

	template<bool inv>
	static void divide_self(topo_acc m, acc<rw> z, acc<ro> x) {
		forall(dof, util::dofs(m), "divide_self") {
			if constexpr (inv) {
				z[dof] = z[dof] / x[dof];
			}
			else {
				z[dof] = x[dof] / z[dof];
			}
		};
	}

	static void divide(topo_acc m, acc<wo> z, acc<ro> x, acc<ro> y) {
		forall(dof, util::dofs(m), "divide") { z[dof] = x[dof] / y[dof]; };
	}

	static void reciprocal_self(topo_acc m, acc<rw> x) {
		forall(dof, util::dofs(m), "reci_self") { x[dof] = 1.0 / x[dof]; };
	}

	static void reciprocal(topo_acc m, acc<wo> x, acc<ro> y) {
		forall(dof, util::dofs(m), "recip") { x[dof] = 1.0 / y[dof]; };
	}

	static void linear_sum(topo_acc m,
	                       acc<wo> z,
	                       scalar alpha,
	                       acc<ro> x,
	                       scalar beta,
	                       acc<ro> y) {

		forall(dof, util::dofs(m), "linear_sum") {
			z[dof] = alpha * x[dof] + beta * y[dof];
		};
	}

	template<bool inv>
	static void linear_sum_self(topo_acc m,
	                            acc<rw> z,
	                            acc<ro> x,
	                            scalar alpha,
	                            scalar beta) {
		forall(dof, util::dofs(m), "linear_sum_self") {
			if constexpr (inv) {
				z[dof] = alpha * z[dof] + beta * x[dof];
			}
			else {
				z[dof] = alpha * x[dof] + beta * z[dof];
			}
		};
	}

	static void
	axpy(topo_acc m, acc<wo> z, scalar alpha, acc<ro> x, acc<ro> y) {
		forall(dof, util::dofs(m), "axpy") {
			z[dof] = alpha * x[dof] + y[dof];
		};
	}

	template<bool inv>
	static void axpy_self(topo_acc m, acc<rw> z, acc<ro> x, scalar alpha) {
		forall(dof, util::dofs(m), "axpy_self") {
			if constexpr (inv) {
				z[dof] = alpha * z[dof] + x[dof];
			}
			else {
				z[dof] = alpha * x[dof] + z[dof];
			}
		};
	}

	static void
	axpby(topo_acc m, acc<rw> y, acc<ro> x, scalar alpha, scalar beta) {
		forall(dof, util::dofs(m), "axpby") {
			y[dof] = alpha * x[dof] + beta * y[dof];
		};
	}

	static void abs_self(topo_acc m, acc<rw> x) {
		forall(dof, util::dofs(m), "abs_self") { x[dof] = std::abs(x[dof]); };
	}

	static void abs(topo_acc m, acc<wo> y, acc<ro> x) {
		forall(dof, util::dofs(m), "abs") { y[dof] = std::abs(x[dof]); };
	}

	static void add_scalar_self(topo_acc m, acc<rw> x, scalar alpha) {
		forall(dof, util::dofs(m), "add_scalar_self") { x[dof] += alpha; };
	}

	static void add_scalar(topo_acc m, acc<wo> y, acc<ro> x, scalar alpha) {
		forall(dof, util::dofs(m), "add_scalar") { y[dof] = x[dof] + alpha; };
	}

	static real lp_norm_local(topo_acc m, acc<ro> u, int p) {
		real ret = reduceall(dof,
		                     up,
		                     util::dofs(m),
		                     flecsi::exec::fold::sum,
		                     real,
		                     "lp_norm_local") {
			up(std::pow(u[dof], p));
		};

		return ret;
	}

	static real l1_norm_local(topo_acc m, acc<ro> u) {
		real ret = reduceall(
			dof, up, util::dofs(m), flecsi::exec::fold::sum, real, "l1_norm") {
			up(std::abs(u[dof]));
		};

		return ret;
	}

	static real l2_norm_local(topo_acc m, acc<ro> u) {
		auto ret = scalar_prod(m, u, u);
		if constexpr (is_complex)
			return ret.real();
		else
			return ret;
	}

	static real local_max(topo_acc m, acc<ro> u) {
		if constexpr (is_complex) {
			auto ret = reduceall(dof,
			                     up,
			                     util::dofs(m),
			                     flecsi::exec::fold::max,
			                     real,
			                     "local_max") {
				up(u[dof].real());
			};
			return ret;
		}
		else {
			auto ret = reduceall(dof,
			                     up,
			                     util::dofs(m),
			                     flecsi::exec::fold::max,
			                     real,
			                     "local_max") {
				up(u[dof]);
			};
			return ret;
		}
	}

	static real local_min(topo_acc m, acc<ro> u) {
		if constexpr (is_complex) {
			auto ret = reduceall(dof,
			                     up,
			                     util::dofs(m),
			                     flecsi::exec::fold::min,
			                     real,
			                     "local-min") {
				up(u[dof].real());
			};
			return ret;
		}
		else {
			auto ret = reduceall(dof,
			                     up,
			                     util::dofs(m),
			                     flecsi::exec::fold::min,
			                     real,
			                     "local-min") {
				up(u[dof]);
			};
			return ret;
		}
	}

	static real inf_norm_local(topo_acc m, acc<ro> x) {
		auto ret = reduceall(
			dof, up, util::dofs(m), flecsi::exec::fold::max, real, "inf_norm") {
			up(std::abs(x[dof]));
		};
		return ret;
	}

	static len local_size(topo_acc m) { return util::dofs(m).size(); }

	static void get_local_size(topo_acc m, len * length) {
		*length = util::dofs(m).size();
	}

	static void set_random(topo_acc m, acc<wo> x, unsigned seed) {
		std::mt19937 gen(seed);
		std::uniform_real_distribution<real> dis(0., 1.);
		for (auto dof : util::dofs(m)) {
			if constexpr (is_complex)
				x[dof] = scalar(dis(gen), dis(gen));
			else
				x[dof] = dis(gen);
		};
	}

	static void dump(std::string_view pre, topo_acc m, acc<ro> x) {
		std::string fname{pre};
		fname += "-" + std::to_string(flecsi::process());
		std::ofstream ofile(fname);
		for (auto dof : util::dofs(m)) {
			ofile << x(dof) << '\n';
		}
	}
};

}
#endif
