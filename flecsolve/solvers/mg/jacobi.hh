#ifndef FLECSOLVE_SOLVERS_MG_JACOBI_H
#define FLECSOLVE_SOLVERS_MG_JACOBI_H

#include "flecsolve/matrices/parcsr.hh"

namespace flecsolve::mg {

template<template<class> class storage>
struct jacobi_params {
	op::core<mat::parcsr_op, storage> A;
	float omega;
	std::size_t nrelax;
	using scalar = mat::parcsr_op::scalar;
	using size = mat::parcsr_op::size;

	jacobi_params(op::core<mat::parcsr_op, storage> a,
	              float o,
	              std::size_t n)
		: A(a), omega(o), nrelax(n) {}

	using topo_t = topo::csr<mat::parcsr_op::scalar, mat::parcsr_op::size>;
	static inline const typename topo_t::template vec_def<topo_t::cols> tmpd;
};

template<template<class> class storage>
jacobi_params(op::core<mat::parcsr_op, storage>, float, std::size_t)->jacobi_params<storage>;

template<template<class> class storage>
struct jacobi : op::base<jacobi_params<storage>> {
	using base = op::base<jacobi_params<storage>>;
	using base::params;
	using scalar = typename base::params_t::scalar;
	using size = typename base::params_t::size;

	jacobi(jacobi_params<storage> p) : base(std::move(p)) {}

	template<class D, class R>
	void apply(const D & b, R & x) const {
		auto & p = const_cast<typename base::params_t&>(params);
		for (std::size_t i = 0; i < params.nrelax; ++i) {
			flecsi::execute<relax>(params.omega,
			                       p.A.source().data.topo(),
			                       x.data.ref(),
			                       b.data.ref(),
			                       p.tmpd(x.data.topo()));
		}
	}

	using topo_t = topo::csr<scalar, size>;
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
}

#endif
