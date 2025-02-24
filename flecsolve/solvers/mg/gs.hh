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
#ifndef FLECSOLVE_SOLVERS_MG_GAUSS_SEIDEL_H
#define FLECSOLVE_SOLVERS_MG_GAUSS_SEIDEL_H

#include <functional>
#include <type_traits>

#include "flecsolve/matrices/parcsr.hh"

namespace flecsolve {

namespace mg {

enum class relax_dir { up, down };
template<class scalar, class size>
struct hybrid_gs_params {
	std::reference_wrapper<mat::parcsr<scalar, size>> A;
	std::size_t nrelax;
	relax_dir direction;

	hybrid_gs_params(std::reference_wrapper<mat::parcsr<scalar, size>> a,
	                 std::size_t n,
	                 relax_dir d)
		: A(a), nrelax(n), direction(d) {}
};

template<class scalar, class size>
struct hybrid_gs;

}

namespace op {

template<class scalar, class size>
struct traits<mg::hybrid_gs<scalar, size>> {
	static constexpr auto input_var = variable<anon_var::anonymous>;
	static constexpr auto output_var = variable<anon_var::anonymous>;
	using parameters = mg::hybrid_gs_params<scalar, size>;
};
}

namespace mg {

template<class scalar, class size>
struct hybrid_gs : op::base<hybrid_gs<scalar, size>> {
	using base = op::base<hybrid_gs<scalar, size>>;
	using base::params;

	hybrid_gs(mg::hybrid_gs_params<scalar, size> p) : base(std::move(p)) {}

	template<class D, class R>
	void apply(const vec::base<D> & b, vec::base<R> & x) {
		for (std::size_t i = 0; i < params.nrelax; ++i) {
			flecsi::execute<relax>(params.direction,
			                       params.A.get().data.topo(),
			                       x.data.ref(),
			                       b.data.ref());
		}
	}

	using topo_t = topo::csr<scalar, size>;
	template<flecsi::partition_privilege_t... PP>
	using vec_acc = typename flecsi::field<scalar>::template accessor<PP...>;

	static void relax(relax_dir dir,
	                  typename topo_t::template accessor<flecsi::ro> A,
	                  vec_acc<flecsi::rw, flecsi::ro> xa,
	                  vec_acc<flecsi::ro, flecsi::na> ba) {
		vec::seq_view x{xa.span()};
		auto b = ba.span();

		auto diag = A.diag();
		auto offd = A.offd();

		auto [drowptr, dcolind, dvalues] = diag.rep();
		auto [orowptr, ocolind, ovalues] = offd.rep();

		auto update = [&](size r) {
			scalar diag = 0.;
			scalar rsum = 0.;
			for (size off = drowptr[r]; off < drowptr[r + 1]; ++off) {
				auto c = dcolind[off];
				auto rid = A.global_id(flecsi::topo::id<topo_t::rows>(r));
				auto cid = A.global_id(flecsi::topo::id<topo_t::cols>(c));
				if (rid == cid)
					diag = dvalues[off];
				else
					rsum += dvalues[off] * x[c];
			}
			for (size off = orowptr[r]; off < orowptr[r + 1]; ++off) {
				rsum += ovalues[off] + x[ocolind[off]];
			}
			auto dinv = 1. / diag;
			x[r] = dinv * (b[r] - rsum);
		};

		if (dir == relax_dir::down) {
			for (size r = 0; r < diag.rows(); ++r)
				update(r);
		}
		else {
			for (size r = diag.rows() - 1; r >= 0; --r)
				update(r);
		}
	}
};
}
}

#endif
