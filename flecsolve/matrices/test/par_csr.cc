#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>
#include <flecsi/execution.hh>

#include "flecsolve/vectors/seq.hh"
#include "flecsolve/matrices/par.hh"
#include "flecsolve/operators/base.hh"
#include "flecsolve/solvers/factory.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/krylov_operator.hh"
#include "flecsolve/vectors/mesh.hh"


namespace flecsolve {

using namespace flecsi;
inline mat::par::slot A;
inline mat::par::cslot coloring;

template<class T, mat::par::index_space ispace>
using matdef = typename field<T>::template definition<mat::par, ispace>;

const matdef<double, mat::par::cols> xdef;
const matdef<double, mat::par::cols> ydef;
const matdef<double, mat::par::cols> tmp;


void spmv_local(field<double>::accessor<wo, wo, na> ya,
                field<double>::accessor<ro, ro, na> xa,
                field<std::size_t>::accessor<ro, na, na> rowptra,
                field<std::size_t>::accessor<ro, na, na> colinda,
                field<double>::accessor<ro, na, na>      valuesa) {
	mat::csr_view A(rowptra.span(), colinda.span(), valuesa.span());

	vec::seq_view y{ya.span()};
	const vec::seq_view x{xa.span()};
	A.mult(x, y);
}


void spmv_remote(field<double>::accessor<wo, wo, na> ya,
                 field<double>::accessor<na, na, ro> xa,
                 field<std::size_t>::accessor<ro, na, na> rowptra,
                 field<std::size_t>::accessor<ro, na, na> colinda,
                 field<double>::accessor<ro, na, na>      valuesa) {
	mat::csr_view A(rowptra.span(), colinda.span(), valuesa.span());

	vec::seq_view y{ya.span()};
	const vec::seq_view x{xa.span()};
	A.mult(x, y);
}

struct parcsr_op : op::base<parcsr_op> {

	static inline const matdef<double, mat::par::cols> tmp;

	template<class D, class R>
	void apply(const D & x, R & y) const {
		flecsi::execute<spmv_remote>(tmp(x.data.topo()), x.data.ref(),
		                             mat::par::rowptr_offd(x.data.topo()),
		                             mat::par::colind_offd(x.data.topo()),
		                             mat::par::values_offd(x.data.topo()));
		flecsi::execute<spmv_local>(y.data.ref(), x.data.ref(),
		                            mat::par::rowptr_diag(x.data.topo()),
		                            mat::par::colind_diag(x.data.topo()),
		                            mat::par::values_diag(x.data.topo()));
		vec::mesh tmpv(x.data.topo(), tmp(x.data.topo()));
		y.add(y, tmpv);
	}
};

int csr_test() {
	UNIT() {
		mat::par::init init_state;
		coloring.allocate("Chem97ZtZ.mtx", init_state);
		A.allocate(coloring.get(), init_state);

		vec::mesh x(A, xdef(A));
		vec::mesh y(A, ydef(A));

		y.set_scalar(0.0);
		x.set_scalar(2);

		op::krylov_parameters params{cg::settings("solver"),
			cg::topo_work<>::get(x),
			parcsr_op{}};
		read_config("parcsr.cfg", params);

		op::krylov slv{std::move(params)};

		auto info = slv.apply(y, x);
		EXPECT_TRUE(info.iters == 167);
	};
	return 0;
}

flecsi::unit::driver<csr_test> driver;
}

