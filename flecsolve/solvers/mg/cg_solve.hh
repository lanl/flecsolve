#ifndef FLECSOLVE_SOLVERS_MG_CG_SOLVE_HH
#define FLECSOLVE_SOLVERS_MG_CG_SOLVE_HH

#include <LapackWrappers.h>

#include "flecsolve/matrices/parcsr.hh"
#include "flecsolve/operators/handle.hh"

namespace flecsolve::mg::ua {

template<class scalar, class size>
struct lapack_solver : op::base<>
{
	using mat_type = mat::parcsr<scalar, size>;
	using op_type = op::core<mat_type>;
	using op_handle = op::handle<op_type>;
	using topo_type = topo::csr<scalar, size>;
	using csr_acc = typename topo::csr<scalar, size>::template accessor<flecsi::ro>;
	template<flecsi::partition_privilege_t priv>
	using vec_acc = typename flecsi::field<scalar>::template accessor<priv, flecsi::na>;

	lapack_solver(op_handle Ah)
		: A(Ah)
	{
		flog_assert(A.get().data.topo().colors() == 1, "lapack solver is serial");
	}

	template<class D, class R>
	void apply(const D & b, R & x) const {
		flecsi::execute<solve_task>(A.get().data.topo(),
		                            x.data.ref(),
		                            b.data.ref());
	}

	static void solve_task(csr_acc mat,
	                       vec_acc<flecsi::wo> x,
	                       vec_acc<flecsi::ro> b) {
		auto diag = mat.diag();
		auto [rowptr, colind, values] = diag.rep();
		int N = rowptr.size() - 1;
		auto copyvec = [=](auto spn) {
			std::vector<double> vec;
			vec.reserve(N);
			std::copy(spn.begin(), spn.begin() + N,
			          std::back_inserter(vec));
			return vec;
		};
		auto bvec = copyvec(b.span());

		std::vector<int> ipiv(N);
		std::vector<double> A(N*N);

		for (size r = 0; r < rowptr.size() - 1; ++r) {
			auto rid = mat.global_id(flecsi::topo::id<topo_type::rows>(r));
			for (size off = rowptr[r]; off < rowptr[r+1]; ++off) {
				auto cid = mat.global_id(flecsi::topo::id<topo_type::cols>(colind[off]));
				A[rid * N + cid] = values[off];
			}
		}
		int err;
		Lapack<scalar>::gesv(N, 1, A.data(), N, ipiv.data(), bvec.data(), N, err);
		flog_assert(err == 0, "LU factorization failed");
		{
			auto xsp = x.span();
			int i{0};
			for (auto v : bvec) xsp[i++] = v;
		}
	}

	op_handle A;
};

}
#endif
