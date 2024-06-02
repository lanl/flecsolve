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
#ifndef FLECSOLVE_SOLVERS_MG_INTERGRID_H
#define FLECSOLVE_SOLVERS_MG_INTERGRID_H

#include "flecsolve/topo/csr.hh"
#include "flecsolve/operators/core.hh"
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
struct prolong : op::base<intergrid_params<scalar, size>> {
	using base = op::base<intergrid_params<scalar, size>>;
	using base::params;
	using parameters = typename base::params_t;
	prolong(parameters p) : base(std::move(p)) {}

	template<class D, class R>
	void apply(const D & x, R & y) const {
		auto & topof = y.data.topo();
		auto & topoc = x.data.topo();
		if (topof.colors() != topoc.colors()) {
			auto lm = flecsi::data::launch::make(topof,
			                                     flecsi::data::launch::block(
				                                     topof.colors(), topoc.colors()));
			using namespace flecsi::data;
			flecsi::execute<injectm>(lm, topoc,
			                         multi_reference(params.aggregates, lm),
			                         x.data.ref(),
			                         multi_reference(y.data.ref(), lm));
		} else {
			flecsi::execute<inject>(y.data.topo(),
			                        x.data.topo(),
			                        params.aggregates,
			                        x.data.ref(),
			                        y.data.ref());
		}
	}

protected:
	using topo_acc =
		typename topo::csr<scalar, size>::template accessor<flecsi::ro>;
	template<flecsi::partition_privilege_t priv>
	using vec_acc =
		typename flecsi::field<scalar>::template accessor<priv, flecsi::na>;
	using aggt_acc = flecsi::field<flecsi::util::id>::accessor<flecsi::ro, flecsi::na>;
	static void inject(
		topo_acc Af,
		topo_acc Ac,
		aggt_acc agg,
		vec_acc<flecsi::ro> x,
		vec_acc<flecsi::wo> y) {
		std::size_t off = Ac.meta().rows.beg;
		std::size_t ncol = Af.meta().cols.size();
		for (std::size_t i{0}; i < ncol; ++i) {
			if (agg[i] != std::numeric_limits<flecsi::util::id>::max())
				y[i] = x[agg[i] - off];
		}
	}

	static void injectm(
		flecsi::data::multi<topo_acc> Af,
		topo_acc Ac,
		flecsi::data::multi<aggt_acc> aggta,
		vec_acc<flecsi::ro> x,
		flecsi::data::multi<vec_acc<flecsi::wo>> ya) {
		std::size_t off = Ac.meta().rows.beg;
		auto aggts = aggta.accessors().begin();
		auto yas = ya.accessors().begin();

		for (auto fine : Af.accessors()) {
			auto aggt = *aggts++;
			auto y = *yas++;
			for (std::size_t i{0}; i < fine.meta().cols.size(); ++i) {
				if (aggt[i] != std::numeric_limits<flecsi::util::id>::max())
					y[i] = x[aggt[i] - off];
			}
		}
	}
};

template<class scalar, class size>
struct restrict : op::base<intergrid_params<scalar, size>> {
	using base = op::base<intergrid_params<scalar, size>>;
	using base::params;
	using parameters = typename base::params_t;
	restrict(parameters p) : base(std::move(p)) {}

	template<class D, class R>
	void apply(const D & x, R & y) const {
		auto & topof = x.data.topo();
		auto & topoc = y.data.topo();
		if (topof.colors() != topoc.colors()) {
				auto lm = flecsi::data::launch::make(topof,
				                                     flecsi::data::launch::block(
					                                     topof.colors(), topoc.colors()));
				using namespace flecsi::data;
				flecsi::execute<avem>(lm, topoc,
				                      multi_reference(params.aggregates, lm),
				                      multi_reference(x.data.ref(), lm),
				                      y.data.ref());
		} else {
			flecsi::execute<ave>(x.data.topo(),
			                     y.data.topo(),
			                     params.aggregates,
			                     x.data.ref(),
			                     y.data.ref());
		}
	}

protected:
	using topo_acc =
		typename topo::csr<scalar, size>::template accessor<flecsi::ro>;
	template<flecsi::partition_privilege_t priv>
	using vec_acc =
		typename flecsi::field<scalar>::template accessor<priv, flecsi::na>;
	using aggt_acc = flecsi::field<flecsi::util::id>::accessor<flecsi::ro, flecsi::na>;
	static void ave(
		topo_acc Af,
		topo_acc Ac,
		aggt_acc agg,
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

	static void avem(flecsi::data::multi<topo_acc> Af,
	                 topo_acc Ac,
	                 flecsi::data::multi<aggt_acc> aggta,
	                 flecsi::data::multi<vec_acc<flecsi::ro>> xa,
	                 vec_acc<flecsi::wo> y) {
		std::fill(y.span().begin(), y.span().end(), 0);
		std::size_t off = Ac.meta().rows.beg;
		auto aggts = aggta.accessors().begin();
		auto xas = xa.accessors().begin();
		for (auto fine : Af.accessors()) {
			auto aggt = *aggts++;
			auto x = *xas++;
			for (std::size_t i{0}; i < fine.meta().cols.size(); ++i) {
				if (aggt[i] != std::numeric_limits<flecsi::util::id>::max()) {
					y[aggt[i] - off] += x[i];
				}
			}
		}
	}
};

}
}

#endif
