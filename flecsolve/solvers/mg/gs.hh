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
#ifndef FLECSOLVE_SOLVERS_MG_GAUSS_SEIDEL_HH
#define FLECSOLVE_SOLVERS_MG_GAUSS_SEIDEL_HH

#include "flecsolve/util/config.hh"
#include "flecsolve/matrices/parcsr.hh"
#include "flecsolve/operators/handle.hh"

#include "flecsolve/matrices/parcsr.hh"

namespace flecsolve::mg {

enum class relax_sweep { forward, backward, symmetric };
enum class relax_dir { forward, backward };

struct hybrid_gs_settings {
	std::size_t nrelax;
	relax_sweep sweep;
};

template<class scalar, class size>
struct bound_hybrid_gs : op::base<> {
	using op_t = op::core<mat::parcsr<scalar, size>>;

	op::handle<op_t> A;

	using topo_t = typename mat::parcsr<scalar, size>::topo_t;

	hybrid_gs_settings settings;

	bound_hybrid_gs(op::handle<op_t> h,
	                const hybrid_gs_settings & s) :
		A(h), settings(s) {}

	template<class D, class R>
	void apply(const D & b, R & x) const {
		auto run = [&](relax_dir rdir) {
			flecsi::execute<relax>(rdir,
			                       A.get().data.topo(),
			                       x.data.ref(),
			                       b.data.ref());
		};
		for (std::size_t i = 0; i < settings.nrelax; ++i) {
			switch (settings.sweep) {
			case relax_sweep::forward:
				run(relax_dir::forward);
				break;
			case relax_sweep::backward:
				run(relax_dir::backward);
				break;
			case relax_sweep::symmetric:
				run(relax_dir::forward);
				run(relax_dir::backward);
				break;
			}
		}
	}

	template<flecsi::partition_privilege_t... PP>
	using vec_acc = typename flecsi::field<scalar>::template accessor<PP...>;

	static void relax(relax_dir rdir,
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
				rsum += ovalues[off] * x[ocolind[off]];
			}
			auto dinv = 1. / diag;
			x[r] = dinv * (b[r] - rsum);
		};

		flecsi::util::iota_view<size> forward(0, diag.rows());
		flecsi::util::transform_view backward(forward,
		                                      [&](size i) { return diag.rows() - i - 1; });
		switch (rdir) {
		case relax_dir::forward:
			for (auto r : forward) update(r);
			break;
		case relax_dir::backward:
			for (auto r : backward) update(r);
			break;
		}
	}
};

struct hybrid_gs {
	using settings = hybrid_gs_settings;

	template<class scalar, class size>
	auto operator()(op::handle<op::core<mat::parcsr<scalar, size>>> A) {
		return op::core<bound_hybrid_gs<scalar, size>>{A, settings_};
	}

	settings settings_;
};

}

#endif
