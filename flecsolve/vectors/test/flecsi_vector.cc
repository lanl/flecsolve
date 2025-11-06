#include <complex>

#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsolve/vectors/topo_view.hh"

#include "flecsolve/util/test/mesh.hh"

using namespace flecsi;

namespace flecsolve {

using realf = field<double>;
using compf = field<std::complex<double>>;

const realf::definition<testmesh, testmesh::cells> xd, yd, zd, tmpd;
const compf::definition<testmesh, testmesh::cells> xd_c, yd_c, zd_c, tmpd_c;

template<int index>
static constexpr double rconv(double gid) {
	return (index + 1) * gid;
}

template<int index>
static constexpr std::complex<double> cconv(double gid) {
	if constexpr (index == 0)
		return {.3 * gid, 0.7 * gid};
	else if constexpr (index == 1)
		return {.1 * gid, .8 * gid};
	else
		return {.5 * gid, .4 * gid};
}

template<class T>
static constexpr std::complex<double> conv(T c) {
	return c;
}

void init_fields(testmesh::accessor<ro, ro> m,
                 realf::accessor<wo, na> xa,
                 realf::accessor<wo, na> ya,
                 realf::accessor<wo, na> za,
                 compf::accessor<wo, na> xa_c,
                 compf::accessor<wo, na> ya_c,
                 compf::accessor<wo, na> za_c) {
	for (auto dof : m.dofs<testmesh::cells>()) {
		double gid = m.global_id(dof);
		xa[dof] = rconv<0>(gid);
		ya[dof] = rconv<1>(gid);
		za[dof] = rconv<2>(gid);

		xa_c[dof] = cconv<0>(gid);
		ya_c[dof] = cconv<1>(gid);
		za_c[dof] = cconv<2>(gid);
	}
}
using namespace std::complex_literals;
template<class P>
struct expected {
	template<bool is_complex>
	using scalar_type = std::conditional_t<is_complex, std::complex<double>, double>;
	static constexpr double ftol = 1e-8;

	static int real(flecsi::exec::accelerator s,
					testmesh::accessor<ro, ro> m,
					realf::accessor<ro, na> x) noexcept {
		UNIT(P::name) {
			auto res = s.executor().reduceall(dof, up,
											  m.dofs<testmesh::cells>(),
											  flecsi::exec::fold::sum,
											  double) {
				auto gid = m.global_id(dof);
				up(std::abs(P::real_answer(gid) - x[dof]));
			};
			EXPECT_LT(res, m.dofs<testmesh::cells>().size() * ftol);
		};
	}

	static int complex(flecsi::exec::cpu s,
					   testmesh::accessor<ro, ro> m,
					   compf::accessor<ro, na> x) noexcept {
		UNIT(P::name) {
			double diff = 0;
			for (auto dof : m.dofs<testmesh::cells>()) {
				auto gid = m.global_id(dof);
				diff += std::abs(P::complex_answer(gid) - x[dof]);
			}
			EXPECT_LT(diff, m.dofs<testmesh::cells>().size() * ftol);
		};
	}

	template<bool is_complex>
	static auto abs_error(scalar_type<is_complex> got) {
		return std::abs(P::template answer<is_complex>() - got);
	}
};
struct add_check : expected<add_check> {
	static constexpr const char * name = "add";
	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return rconv<0>(gid) + rconv<2>(gid);
	}
	static auto complex_answer(double gid) {
		return cconv<0>(gid) + cconv<2>(gid);
	}
};

struct sub_check : expected<sub_check> {
	static constexpr const char * name = "subtract";
	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return rconv<0>(gid) - rconv<2>(gid);
	}
	static auto complex_answer(double gid) {
		return cconv<0>(gid) - cconv<2>(gid);
	}
};

struct mult_check : expected<mult_check> {
	static constexpr const char * name = "multiply";
	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return rconv<0>(gid) * rconv<2>(gid);
	}
	static auto complex_answer(double gid) {
		return cconv<0>(gid) * cconv<2>(gid);
	}
};

struct scalar_add_check : expected<scalar_add_check> {
	static constexpr const char * name = "add scalar";
	template<bool is_complex>
	FLECSI_INLINE_TARGET static auto scalar_value() {
		if constexpr (is_complex) return 1. + 1i;
		else return 1;
	}

	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return rconv<0>(gid) + scalar_value<false>();
	}
	static auto complex_answer(double gid) {
		return cconv<0>(gid) + conv(scalar_value<true>());
	}
};

struct div_check : expected<div_check> {
	static constexpr const char * name = "divide";
	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return rconv<1>(gid) / (rconv<0>(gid) + scalar_add_check::scalar_value<false>());
	}
	static auto complex_answer(double gid) {
		return cconv<1>(gid) /
			(cconv<0>(gid) + conv(scalar_add_check::scalar_value<true>()));
	}
};

struct scale_check : expected<scale_check> {
	template<bool is_complex>
	FLECSI_INLINE_TARGET static auto shift_value() {
		if constexpr (!is_complex) return -1;
		else return -1. - 1i;
	}

	template<bool is_complex>
	FLECSI_INLINE_TARGET static double scale_value() {
		if constexpr (!is_complex) return 2;
		else return 2.4;
	}

	static constexpr const char * name = "scale";
	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return rconv<0>(gid) * scale_value<false>();
	}
	static auto complex_answer(double gid) {
		return cconv<0>(gid) * scale_value<true>();
	}
};

struct recip_check : expected<recip_check> {
	static constexpr const char * name = "reciprocal";

	template<bool is_complex>
	FLECSI_INLINE_TARGET static auto shift_value() {
		if constexpr (!is_complex) return 1;
		else return 1. + 1i;
	}

	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return 1.0 / (rconv<1>(gid) + shift_value<false>());
	}
	static auto complex_answer(double gid) {
		return 1.0 / (cconv<1>(gid) + conv(shift_value<true>()));
	}
};

struct linsum_check : expected<linsum_check> {
	static constexpr const char * name = "linear sum";
	template<bool is_complex>
	FLECSI_INLINE_TARGET static auto shift_value() {
		if constexpr (!is_complex) return -1;
		else return -1. - 1i;
	}
	inline static constexpr double alpha = 8;
	inline static constexpr double beta = 9;
	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return rconv<1>(gid) * alpha + rconv<2>(gid) * beta;
	}
	static auto complex_answer(double gid) {
		return cconv<1>(gid) * alpha + cconv<2>(gid) * beta;
	}
};

struct axpy_check : expected<axpy_check> {
	static constexpr const char * name = "axpy";
	template<bool is_complex>
	FLECSI_INLINE_TARGET static auto alpha() {
		if constexpr (!is_complex) return 7;
		else return 7. + 3i;
	}

	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return rconv<0>(gid) * alpha<false>() + rconv<1>(gid);
	}
	static auto complex_answer(double gid) {
		return cconv<0>(gid) * conv(alpha<true>()) + cconv<1>(gid);
	}
};

struct axpby_check : expected<axpby_check> {
	static constexpr const char * name = "axpby";
	template<bool is_complex>
	FLECSI_INLINE_TARGET static auto alpha() {
		if constexpr (!is_complex) return 4;
		else return 4.3 + 7i;
	}

	template<bool is_complex>
	FLECSI_INLINE_TARGET static auto beta() {
		if constexpr (!is_complex) return 11;
		else return 11.8 + 3i;
	}

	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return rconv<2>(gid) * alpha<false>() + rconv<1>(gid) * beta<false>();
	}

	static auto complex_answer(double gid) {
		return cconv<2>(gid) * conv(alpha<true>()) +
			cconv<1>(gid) * conv(beta<true>());
	}
};

struct abs_check : expected<abs_check> {
	static constexpr const char * name = "abs";
	template<bool is_complex>
	FLECSI_INLINE_TARGET static auto shift() {
		if constexpr (!is_complex) return -4;
		else return -4. -4i;
	}

	FLECSI_INLINE_TARGET static double real_answer(double gid) {
		return std::abs(rconv<1>(gid) + shift<false>());
	}

	inline static auto complex_answer(double gid) {
		return std::abs(cconv<1>(gid) + conv(shift<true>()));
	}
};

struct dot_check : expected<dot_check> {
	template<bool is_complex>
	static auto answer() {
		auto [alpha, beta] = scalars<is_complex>();
		return scalar_type<is_complex>(32) * alpha * (is_complex ? std::conj(beta) : beta);
	}

	template<bool is_complex>
	static auto scalars() {
		if constexpr (is_complex) return std::pair{1.2 + 3i, 7. + 2.3i};
		else return std::pair{1.5, 3.8};
	}
};

struct l1norm_check : expected<l1norm_check> {
	template<bool is_complex>
	static auto scalar() {
		if constexpr (is_complex) return 1.3 + 8.7i;
		else return -3.141719;
	}
	template<bool is_complex>
	static double answer() {
		return 32 * std::abs(scalar<is_complex>());
	}
};

struct l2norm_check : expected<l2norm_check> {
	template<bool is_complex>
	static auto scalar() {
		return l1norm_check::scalar<is_complex>();
	}

	template<bool is_complex>
	static auto answer() {
		return std::sqrt(scalar_type<is_complex>(32) *
						 scalar<is_complex>() *
						 (is_complex ? std::conj(scalar<is_complex>()) : scalar<is_complex>()));
	}
};

int vectest(flecsi::scheduler & s) {
	testmesh::ptr mptr;

	auto & msh = init_mesh(s, 32, mptr);
	execute<init_fields>(
		msh, xd(msh), yd(msh), zd(msh), xd_c(msh), yd_c(msh), zd_c(msh));

	UNIT () {
		auto create = [&](auto &... defs) {
			return std::tuple(vec::make(defs(msh))...);
		};

		auto [x, y, z, tmp] = create(xd, yd, zd, tmpd);
		auto [x_c, y_c, z_c, tmp_c] = create(xd_c, yd_c, zd_c, tmpd_c);

		auto test_vecops = [&](auto & x, auto & y, auto & z, auto & tmp) -> int {
			constexpr bool is_complex = num_traits<typename std::decay_t<decltype(x)>::scalar>::is_complex;
			auto check_answer = [&](auto expect, auto & vec) {
				if constexpr (is_complex) {
					EXPECT_EQ(s.test<decltype(expect)::complex>(
								  flecsi::exec::on, msh, vec.data.ref()),
					          0);
				} else {
					EXPECT_EQ(s.test<decltype(expect)::real>(
								  flecsi::exec::on, msh, vec.data.ref()),
					          0);
				}
			};

			tmp.add(x, z);
			check_answer(add_check{}, tmp);
			tmp.subtract(x, z);
			check_answer(sub_check{}, tmp);

			tmp.multiply(x, z);
			check_answer(mult_check{}, tmp);

			x.add_scalar(x, scalar_add_check::scalar_value<is_complex>());
			check_answer(scalar_add_check{}, x);

			tmp.divide(y, x);
			check_answer(div_check{}, tmp);

			x.add_scalar(x, scale_check::shift_value<is_complex>());
			tmp.scale(scale_check::scale_value<is_complex>(), x);
			check_answer(scale_check{}, tmp);

			y.add_scalar(y, recip_check::shift_value<is_complex>());
			tmp.reciprocal(y);
			check_answer(recip_check{}, tmp);

			y.add_scalar(y, linsum_check::shift_value<is_complex>());
			tmp.linear_sum(linsum_check::alpha, y, linsum_check::beta, z);
			check_answer(linsum_check{}, tmp);

			tmp.axpy(axpy_check::alpha<is_complex>(), x, y);
			check_answer(axpy_check{}, tmp);

			tmp.copy(y);
			tmp.axpby(axpby_check::alpha<is_complex>(), axpby_check::beta<is_complex>(), z);
			check_answer(axpby_check{}, tmp);

			tmp.add_scalar(y, abs_check::shift<is_complex>());
			tmp.abs(tmp);
			check_answer(abs_check{}, tmp);

			if constexpr (!is_complex) {
				tmp.add_scalar(y, -7);
				EXPECT_EQ(tmp.min().get(), -7);

				EXPECT_EQ(z.max().get(), 93);
			}

			auto test_reductions = [&](auto & a, auto & b) {
				auto [alpha, beta] = dot_check::scalars<is_complex>();
				a.set_scalar(alpha);
				b.set_scalar(beta);
				EXPECT_LT(dot_check::abs_error<is_complex>(a.dot(b).get()), dot_check::ftol);

				a.set_scalar(l1norm_check::scalar<is_complex>());
				EXPECT_LT(l1norm_check::abs_error<is_complex>(a.l1norm().get()), l1norm_check::ftol);
				EXPECT_LT(l2norm_check::abs_error<is_complex>(a.l2norm().get()), l2norm_check::ftol);
			};
			test_reductions(x, y);

			return 0;
		};

		EXPECT_EQ(test_vecops(x, y, z, tmp), 0);
#ifndef KOKKOS_ENABLE_CUDA
		EXPECT_EQ(test_vecops(x_c, y_c, z_c, tmp_c), 0);
#endif
	};
}

flecsi::util::unit::driver<vectest> driver;

}
