#ifndef FLECSOLVE_MATRICES_PARCSR_H
#define FLECSOLVE_MATRICES_PARCSR_H

#include "flecsolve/topo/csr.hh"
#include "flecsolve/vectors/mesh.hh"

namespace flecsolve::mat {

template<class scalar, class size>
struct parcsr;
template<class scalar, class size>
struct traits<parcsr<scalar, size>> {
	using scalar_t = scalar;
	using size_t = size;
	using topo_t = topo::csr<scalar, size>;
	struct data_t {
		typename topo_t::slot & topo() {
			if (!topo_slot)
				topo_slot = std::make_unique<typename topo_t::slot>();
			return *topo_slot;
		}
		typename topo_t::cslot coloring;
		typename topo_t::init coloring_input;

		auto spmv_tmp() { return vec::mesh(topo(), spmv_tmp_def(topo())); }

	protected:
		typename flecsi::field<scalar_t>::template definition<topo_t,
		                                                      topo_t::cols>
			spmv_tmp_def;
		std::unique_ptr<typename topo_t::slot> topo_slot;
	};

	struct ops_t {
		template<class D, class R>
		void spmv(const D & x, const data_t & data_c, R & y) const {
			auto & data = const_cast<data_t &>(data_c);
			auto tmpv = const_cast<data_t &>(data).spmv_tmp();
			flecsi::execute<spmv_remote>(
				data.topo(), tmpv.data.ref(), x.data.ref());
			flecsi::execute<spmv_local>(
				data.topo(), y.data.ref(), x.data.ref());
			y.add(y, tmpv);
		}

	protected:
		static void spmv_remote(
			typename topo_t::template accessor<flecsi::ro> ma,
			typename flecsi::field<scalar_t>::template accessor<flecsi::wo,
		                                                        flecsi::na> ya,
			typename flecsi::field<
				scalar_t>::template accessor<flecsi::na, flecsi::ro> xa) {
			vec::seq_view y{ya.span()};
			vec::seq_view x{xa.span()};
			ma.offd().mult(x, y);
		}

		static void spmv_local(
			typename topo_t::template accessor<flecsi::ro> ma,
			typename flecsi::field<scalar_t>::template accessor<flecsi::wo,
		                                                        flecsi::na> ya,
			typename flecsi::field<
				scalar_t>::template accessor<flecsi::ro, flecsi::na> xa) {
			vec::seq_view y{ya.span()};
			vec::seq_view x{xa.span()};
			ma.diag().mult(x, y);
		}
	};
};

template<class scalar, class size = std::size_t>
struct parcsr : sparse<parcsr<scalar, size>> {
	using sparse<parcsr<scalar, size>>::data;
	template<typename topo::csr<scalar, size>::index_space S>
	auto vec(typename flecsi::field<
			 scalar>::template definition<topo::csr<scalar, size>, S> & def) {
		return vec::mesh(data.topo(), def(data.topo()));
	}
};

}

#endif
