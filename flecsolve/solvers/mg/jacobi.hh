#ifndef FLECSOLVE_SOLVERS_MG_JACOBI_H
#define FLECSOLVE_SOLVERS_MG_JACOBI_H

#include "flecsolve/matrices/parcsr.hh"

namespace flecsolve {

namespace mg {
template<class scalar, class size>
struct jacobi_params {
	std::reference_wrapper<mat::parcsr<scalar, size>> A;
	float omega;
	std::size_t nrelax;

	jacobi_params(std::reference_wrapper<mat::parcsr<scalar, size>> a,
	              float o,
	              std::size_t n)
		: A(a), omega(o), nrelax(n) {}

	using topo_t = topo::csr<scalar, size>;
	static inline const typename topo_t::template vec_def<topo_t::cols> tmpd;
};

template<class scalar, class size>
struct jacobi;
}

namespace op {
template<class scalar, class size>
struct traits<mg::jacobi<scalar, size>> {
	static constexpr auto input_var = variable<anon_var::anonymous>;
	static constexpr auto output_var = variable<anon_var::anonymous>;
	using parameters = mg::jacobi_params<scalar, size>;
};
}

namespace mg {

template<class scalar, class size>
struct jacobi : op::base<jacobi<scalar, size>> {
	using base = op::base<jacobi<scalar, size>>;
	using base::params;
	jacobi(mg::jacobi_params<scalar, size> p) : base(std::move(p)) {}

	template<class D, class R>
	void apply(const vec::base<D> & b, vec::base<R> & x) {
		for (std::size_t i = 0; i < params.nrelax; ++i) {
			flecsi::execute<relax>(params.omega,
			                       params.A.get().data.topo(),
			                       x.data.ref(),
			                       b.data.ref(),
			                       params.tmpd(x.data.topo()));
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

		auto drowptr = diag.offsets();
		auto dcolind = diag.indices();
		auto dvalues = diag.values();

		auto orowptr = offd.offsets();
		auto ocolind = offd.indices();
		auto ovalues = offd.values();

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
}

#endif
