#pragma once

#include <cmath>
#include <random>

#include <flecsi/execution.hh>

#include "flecsi-linalg/util/traits.hh"
#include "flecsi-linalg/vectors/data/mesh.hh"

namespace flecsi::linalg::vec::ops {

template<class VecData, class Scalar, class Len>
struct mesh_tasks {

	using scalar = Scalar;
	using real = typename num_traits<Scalar>::real;
	using len = Len;
	static constexpr bool is_complex = num_traits<scalar>::is_complex;
	using topo_acc = typename VecData::topo_acc;

	template<partition_privilege_t priv>
	using acc = typename VecData::template acc<priv>;

	template<partition_privilege_t priv>
	using acc_all = typename VecData::template acc_all<priv>;

	using util = typename VecData::util;

	static scalar scalar_prod(topo_acc m, acc<ro> x, acc<ro> y) {
		scalar res = 0.0;

		if constexpr (is_complex) {
			for (auto dof : util::dofs(m)) {
				res += std::conj(y[dof]) * x[dof];
			}
		}
		else {
			for (auto dof : util::dofs(m)) {
				res += x[dof] * y[dof];
			}
		}

		return res;
	}

	static void set_to_scalar(topo_acc m, acc<wo> x, scalar val) {
		for (auto dof : util::dofs(m)) {
			x[dof] = val;
		}
	}

	static void scale_self(topo_acc m, acc<rw> x, scalar val) {
		for (auto dof : util::dofs(m)) {
			x[dof] *= val;
		}
	}

	static void scale(topo_acc m, acc<ro> x, acc<wo> y, scalar val) {
		for (auto dof : util::dofs(m)) {
			y[dof] = x[dof] * val;
		}
	}

	template<class OtherAcc>
	static void copy(topo_acc m, acc_all<wo> xa, OtherAcc ya) {
		const auto in = ya.span();
		auto out = xa.span();
		std::copy(in.begin(), in.end(), out.begin());
	}

	static void add_self(topo_acc m, acc<wo> z, acc<ro> x) {
		for (auto dof : util::dofs(m)) {
			z[dof] = z[dof] + x[dof];
		}
	}

	static void add(topo_acc m, acc<wo> z, acc<ro> x, acc<ro> y) {
		for (auto dof : util::dofs(m)) {
			z[dof] = x[dof] + y[dof];
		}
	}

	static void subtract(topo_acc m, acc<wo> x, acc<ro> a, acc<ro> b) {
		for (auto dof : util::dofs(m)) {
			x[dof] = a[dof] - b[dof];
		}
	}

	template<bool inv>
	static void subtract_self(topo_acc m, acc<wo> x, acc<ro> b) {
		for (auto dof : util::dofs(m)) {
			if constexpr (inv) {
				x[dof] = x[dof] - b[dof];
			}
			else {
				x[dof] = b[dof] - x[dof];
			}
		}
	}

	static void multiply(topo_acc m, acc<wo> z, acc<ro> x, acc<ro> y) {
		for (auto dof : util::dofs(m)) {
			z[dof] = x[dof] * y[dof];
		}
	}

	static void multiply_self(topo_acc m, acc<rw> x, acc<ro> y) {
		for (auto dof : util::dofs(m)) {
			x[dof] = x[dof] * y[dof];
		}
	}

	template<bool inv>
	static void divide_self(topo_acc m, acc<rw> z, acc<ro> x) {
		for (auto dof : util::dofs(m)) {
			if constexpr (inv) {
				z[dof] = z[dof] / x[dof];
			}
			else {
				z[dof] = x[dof] / z[dof];
			}
		}
	}

	static void divide(topo_acc m, acc<wo> z, acc<ro> x, acc<ro> y) {
		for (auto dof : util::dofs(m)) {
			z[dof] = x[dof] / y[dof];
		}
	}

	static void reciprocal_self(topo_acc m, acc<rw> x) {
		for (auto dof : util::dofs(m)) {
			x[dof] = 1.0 / x[dof];
		}
	}

	static void reciprocal(topo_acc m, acc<wo> x, acc<ro> y) {
		for (auto dof : util::dofs(m)) {
			x[dof] = 1.0 / y[dof];
		}
	}

	static void linear_sum(topo_acc m,
	                       acc<wo> z,
	                       scalar alpha,
	                       acc<ro> x,
	                       scalar beta,
	                       acc<ro> y) {

		for (auto dof : util::dofs(m)) {
			z[dof] = alpha * x[dof] + beta * y[dof];
		}
	}

	template<bool inv>
	static void linear_sum_self(topo_acc m,
	                            acc<rw> z,
	                            acc<ro> x,
	                            scalar alpha,
	                            scalar beta) {
		for (auto dof : util::dofs(m)) {
			if constexpr (inv) {
				z[dof] = alpha * z[dof] + beta * x[dof];
			}
			else {
				z[dof] = alpha * x[dof] + beta * z[dof];
			}
		}
	}

	static void
	axpy(topo_acc m, acc<wo> z, scalar alpha, acc<ro> x, acc<ro> y) {
		for (auto dof : util::dofs(m)) {
			z[dof] = alpha * x[dof] + y[dof];
		}
	}

	template<bool inv>
	static void axpy_self(topo_acc m, acc<rw> z, acc<ro> x, scalar alpha) {
		for (auto dof : util::dofs(m)) {
			if constexpr (inv) {
				z[dof] = alpha * z[dof] + x[dof];
			}
			else {
				z[dof] = alpha * x[dof] + z[dof];
			}
		}
	}

	static void
	axpby(topo_acc m, acc<rw> y, acc<ro> x, scalar alpha, scalar beta) {
		for (auto dof : util::dofs(m)) {
			y[dof] = alpha * x[dof] + beta * y[dof];
		}
	}

	static void abs_self(topo_acc m, acc<rw> x) {
		for (auto dof : util::dofs(m)) {
			x[dof] = std::abs(x[dof]);
		}
	}

	static void abs(topo_acc m, acc<wo> y, acc<ro> x) {
		for (auto dof : util::dofs(m)) {
			y[dof] = std::abs(x[dof]);
		}
	}

	static void add_scalar_self(topo_acc m, acc<rw> x, scalar alpha) {
		for (auto dof : util::dofs(m)) {
			x[dof] += alpha;
		}
	}

	static void add_scalar(topo_acc m, acc<wo> y, acc<ro> x, scalar alpha) {
		for (auto dof : util::dofs(m)) {
			y[dof] = x[dof] + alpha;
		}
	}

	static real lp_norm_local(topo_acc m, acc<ro> u, int p) {
		real ret = 0;
		for (auto dof : util::dofs(m)) {
			ret += std::pow(u[dof], p);
		}

		return ret;
	}

	static real l1_norm_local(topo_acc m, acc<ro> u) {
		real ret = 0;
		for (auto dof : util::dofs(m)) {
			ret += std::abs(u[dof]);
		}

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
		auto ret = std::numeric_limits<real>::min();
		for (auto dof : util::dofs(m)) {
			if constexpr (is_complex)
				ret = std::max(u[dof].real(), ret);
			else
				ret = std::max(u[dof], ret);
		}
		return ret;
	}

	static real local_min(topo_acc m, acc<ro> u) {
		auto ret = std::numeric_limits<real>::max();
		for (auto dof : util::dofs(m)) {
			if constexpr (is_complex)
				ret = std::min(u[dof].real(), ret);
			else
				ret = std::min(u[dof], ret);
		}
		return ret;
	}

	static real inf_norm_local(topo_acc m, acc<ro> x) {
		auto ret = std::numeric_limits<real>::min();
		for (auto dof : util::dofs(m)) {
			ret = std::max(std::abs(x[dof]), ret);
		}
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
		}
	}

	static void dump(std::string_view pre, topo_acc m, acc<ro> x) {
		std::string fname{pre};
		fname += "-" + std::to_string(process());
		std::ofstream ofile(fname);
		for (auto dof : util::dofs(m)) {
			ofile << x(dof) << '\n';
		}
	}
};

}
