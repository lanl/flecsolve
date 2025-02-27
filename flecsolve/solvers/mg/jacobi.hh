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
#ifndef FLECSOLVE_SOLVERS_MG_JACOBI_H
#define FLECSOLVE_SOLVERS_MG_JACOBI_H

#include "flecsolve/util/config.hh"
#include "flecsolve/matrices/parcsr.hh"
#include "flecsolve/operators/storage.hh"

namespace flecsolve::mg {

struct jacobi_settings {
	float omega;
	std::size_t nrelax;
};

template<class Op>
struct bound_jacobi : op::base<> {
	using store = op::storage<Op>;
	using op_t = typename store::op_type;
	using scalar = typename op_t::scalar_type;
	using size = typename op_t::size_type;
	store A;

	using topo_t = typename mat::parcsr<scalar, size>::topo_t;
	static inline const typename topo_t::template vec_def<topo_t::cols> tmpd;
	jacobi_settings settings;

	template<class O>
	bound_jacobi(O && o,
	             jacobi_settings s)
		: A(std::forward<O>(o)), settings(s) {}

	template<class D, class R>
	void apply(const D & b, R & x) const {
		for (std::size_t i = 0; i < settings.nrelax; ++i) {
			flecsi::execute<relax>(settings.omega,
			                       A.get().data.topo(),
			                       x.data.ref(),
			                       b.data.ref(),
			                       tmpd(x.data.topo()));
		}
	}

	template<flecsi::partition_privilege_t... PP>
	using vec_acc = typename flecsi::field<scalar>::template accessor<PP...>;

	static void relax(scalar omega,
	                  typename topo_t::template accessor<flecsi::ro> A,
	                  vec_acc<flecsi::rw, flecsi::ro> xa,
	                  vec_acc<flecsi::ro, flecsi::na> ba,
	                  vec_acc<flecsi::wo, flecsi::wo> tmpa) {
		std::copy(xa.span().begin(), xa.span().end(), tmpa.span().begin());

		vec::seq_view x{xa.span()};
		auto b = ba.span();
		vec::seq_view tmp{tmpa.span()};
		auto diag = A.diag();
		auto offd = A.offd();

		auto [drowptr, dcolind, dvalues] = diag.rep();
		auto [orowptr, ocolind, ovalues] = offd.rep();

		for (size r = 0; r < diag.rows(); ++r) {
			scalar diag = 0;
			scalar lpu_x = 0;
			for (size off = drowptr[r]; off < drowptr[r + 1]; ++off) {
				auto c = dcolind[off];
				auto rid = A.global_id(flecsi::topo::id<topo_t::rows>(r));
				auto cid = A.global_id(flecsi::topo::id<topo_t::cols>(c));
				if (rid == cid)
					diag = dvalues[off];
				else {
					lpu_x += dvalues[off] * tmp[c];
				}
			}
			for (size off = orowptr[r]; off < orowptr[r + 1]; ++off) {
				lpu_x += ovalues[off] * tmp[ocolind[off]];
			}
			auto dinv = 1. / diag;
			x[r] = omega * dinv * (b[r] - lpu_x) + (1 - omega) * tmp[r];
		}
	}
};

namespace po = boost::program_options;

struct jacobi {
	using settings = jacobi_settings;
	struct options : with_label {
		using settings_type = settings;
		explicit options(const char * pre) : with_label(pre) {}

		auto operator()(settings_type & s) {
			po::options_description desc;
			// clang-format off
			desc.add_options()
				(label("omega").c_str(), po::value<float>(&s.omega)->default_value(2/3.), "Jacobi weight")
				(label("nrelax").c_str(), po::value<std::size_t>(&s.nrelax)->required(), "Number of relaxation steps");
			// clang-format on
			return desc;
		}
	};

	template<class A>
	auto operator()(A && a) {
		return op::core<bound_jacobi<std::decay_t<A>>>{std::forward<A>(a), settings_};
	}

	jacobi_settings settings_;
};
}

#endif
