#ifndef FLECSOLVE_SOLVERS_MG_INTERGRID_H
#define FLECSOLVE_SOLVERS_MG_INTERGRID_H

#include "flecsolve/topo/csr.hh"
#include "flecsolve/operators/base.hh"
#include <limits>

namespace flecsolve {

namespace mg::ua {
template<class scalar, class size>
struct intergrid_params {
	using topo_t = topo::csr<scalar, size>;
	using ref_t =
		flecsi::field<flecsi::util::id>::Reference<topo_t, topo_t::cols>;

	ref_t aggregates; // field representing the transpose of the aggregates
};

template<class scalar, class size>
struct prolong;

template<class scalar, class size>
struct restrict;
}

namespace op {
template<class scalar, class size>
struct traits<mg::ua::prolong<scalar, size>> {
	static constexpr auto input_var = variable<anon_var::anonymous>;
	static constexpr auto output_var = variable<anon_var::anonymous>;
	using parameters = mg::ua::intergrid_params<scalar, size>;
};

template<class scalar, class size>
struct traits<mg::ua::restrict<scalar, size>> {
	static constexpr auto input_var = variable<anon_var::anonymous>;
	static constexpr auto output_var = variable<anon_var::anonymous>;
	using parameters = mg::ua::intergrid_params<scalar, size>;
};

}

namespace mg::ua {

template<class scalar, class size>
struct prolong : op::base<prolong<scalar, size>> {
	using base = op::base<prolong<scalar, size>>;
	using base::params;
	using parameters = typename base::params_type;
	prolong(parameters p) : base(std::move(p)) {}

	template<class D, class R>
	void apply(const D & x, R & y) const {
		flecsi::execute<inject>(y.data.topo(),
		                        x.data.topo(),
		                        params.aggregates,
		                        x.data.ref(),
		                        y.data.ref());
	}

protected:
	using topo_acc =
		typename topo::csr<scalar, size>::template accessor<flecsi::ro>;
	template<flecsi::partition_privilege_t priv>
	using vec_acc =
		typename flecsi::field<scalar>::template accessor<priv, flecsi::na>;
	static void inject(
		topo_acc Af,
		topo_acc Ac,
		flecsi::field<flecsi::util::id>::accessor<flecsi::ro, flecsi::na> agg,
		vec_acc<flecsi::ro> x,
		vec_acc<flecsi::wo> y) {
		std::size_t off = Ac.meta().rows.beg;
		std::size_t ncol = Af.meta().cols.size();
		for (std::size_t i{0}; i < ncol; ++i) {
			if (agg[i] != std::numeric_limits<flecsi::util::id>::max())
				y[i] = x[agg[i] - off];
		}
	}
};

template<class scalar, class size>
struct restrict : op::base<restrict<scalar, size>> {
	using base = op::base<restrict<scalar, size>>;
	using base::params;
	using parameters = typename base::params_type;
	restrict(parameters p) : base(std::move(p)) {}

	template<class D, class R>
	void apply(const D & x, R & y) const {
		flecsi::execute<ave>(x.data.topo(),
		                     y.data.topo(),
		                     params.aggregates,
		                     x.data.ref(),
		                     y.data.ref());
	}

protected:
	using topo_acc =
		typename topo::csr<scalar, size>::template accessor<flecsi::ro>;
	template<flecsi::partition_privilege_t priv>
	using vec_acc =
		typename flecsi::field<scalar>::template accessor<priv, flecsi::na>;
	static void ave(
		topo_acc Af,
		topo_acc Ac,
		flecsi::field<flecsi::util::id>::accessor<flecsi::ro, flecsi::na> agg,
		vec_acc<flecsi::ro> x,
		vec_acc<flecsi::wo> y) {
		std::fill(y.span().begin(), y.span().end(), 0);

		std::size_t off = Ac.meta().rows.beg;
		std::size_t ncol = Af.meta().cols.size();
		for (std::size_t i{0}; i < ncol; ++i) {
			if (agg[i] != std::numeric_limits<flecsi::util::id>::max())
				y[agg[i] - off] += x[i];
		}
	}
};

}
}

#endif
