#ifndef FLECSOLVE_MATRICES_PARCSR_H
#define FLECSOLVE_MATRICES_PARCSR_H

#include "flecsolve/topo/csr.hh"
#include "flecsolve/vectors/mesh.hh"

namespace flecsolve::mat {

struct parcsr;
template<>
struct traits<parcsr> {
	using scalar_t = double;
	using size_t = std::size_t;
	struct data_t {
		topo::csr::slot & topo() {
			if (!topo_slot)
				topo_slot = std::make_unique<topo::csr::slot>();
			return *topo_slot;
		}
		topo::csr::cslot coloring;
		topo::csr::init coloring_input;

		auto spmv_tmp() { return vec::mesh(topo(), spmv_tmp_def(topo())); }

	protected:
		flecsi::field<scalar_t>::definition<topo::csr, topo::csr::cols>
			spmv_tmp_def;
		std::unique_ptr<topo::csr::slot> topo_slot;
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
			topo::csr::accessor<flecsi::ro> ma,
			flecsi::field<scalar_t>::accessor<flecsi::wo, flecsi::na> ya,
			flecsi::field<scalar_t>::accessor<flecsi::na, flecsi::ro> xa) {
			vec::seq_view y{ya.span()};
			vec::seq_view x{xa.span()};
			ma.offd().mult(x, y);
		}

		static void spmv_local(
			topo::csr::accessor<flecsi::ro> ma,
			flecsi::field<scalar_t>::accessor<flecsi::wo, flecsi::na> ya,
			flecsi::field<scalar_t>::accessor<flecsi::ro, flecsi::na> xa) {
			vec::seq_view y{ya.span()};
			vec::seq_view x{xa.span()};
			ma.diag().mult(x, y);
		}
	};
};

struct parcsr : sparse<parcsr> {};

}

#endif
