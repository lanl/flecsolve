#ifndef FLECSI_LINALG_OP_KRYLOV_INTERFACE_H
#define FLECSI_LINALG_OP_KRYLOV_INTERFACE_H

#include <tuple>

#include "factory.hh"

namespace flecsi::linalg {

template <class S, class A, class P, class D>
struct krylov_op {
	S slv;
	A op;
	P precond;
	D diag;

	template<class DomainVec, class RangeVec>
	auto apply(const RangeVec & b, DomainVec & x) {
		return slv.apply(op, b, x, precond, diag);
	}

	template<class T>
	void reset(const T & settings) {
		slv.reset(settings);
	}

	template<class ... Args>
	auto rebind(Args&& ... args) {
		return slv.bind(std::forward<Args>(args)...);
	}
};
template <class S, class A, class P, class D>
krylov_op(S&&,A&&,P&&,D&&)->krylov_op<S,A,P,D>;

template <class Workspace, template<class> class Solver>
struct krylov_interface {
	using workvec_t = typename std::remove_reference_t<Workspace>::value_type;
	using real = typename workvec_t::real;

	template<class A>
	auto bind(A&& op) & {
		return krylov_op{static_cast<Solver<Workspace>&>(*this),
			std::forward<A>(op), op::I, [](const auto&,double){return false;}};
	}

	template<class A>
	auto bind(A&& op) && {
		return krylov_op{
			std::move(*static_cast<Solver<Workspace>*>(this)),
			std::forward<A>(op), op::I, [](const auto&,double){return false;}};
	}


	template<class A, class P>
	auto bind(A&& op, P&& p) & {
		return krylov_op{static_cast<Solver<Workspace>&>(*this),
			std::forward<A>(op), std::forward<P>(p), [](const auto&,double){return false;}};
	}

	template<class A, class P>
	auto bind(A&& op, P&& p) && {
		return krylov_op{
			std::move(*static_cast<Solver<Workspace>*>(this)),
			std::forward<A>(op), std::forward<P>(p), [](const auto&,double){return false;}};
	}


	template<class A, class P, class D>
	auto bind(A&& op, P&& p, D&& d) & {
		return krylov_op{static_cast<Solver<Workspace>&>(*this),
			std::forward<A>(op), std::forward<P>(p), std::forward<D>(d)};
	}

	template<class A, class P, class D>
	auto bind(A&& op, P&& p, D&& d) && {
		return krylov_op{
			std::move(*static_cast<Solver<Workspace>*>(this)),
			std::forward<A>(op), std::forward<P>(p), std::forward<D>(d)};
	}

	Workspace work;
};

template<template<class> class S, class W, class ... Ops>
struct krylov_params {
	using settings_type = typename S<W>::settings_type;
	template<class T>
	using solver_type = S<T>;
	using op_class = krylov_interface<W, S>;

	template<class V, class ... O>
	krylov_params(settings_type s, V&& w, O&& ... o) :
		solver_settings(std::move(s)),
		work(std::forward<V>(w)),
		ops(std::forward<O>(o)...) {}

	settings_type solver_settings;
	W work;
	std::tuple<Ops...> ops;
};


template<template<class> class S, class W, class ... Ops>
auto make_krylov_op(const typename S<W>::settings_type & params, W && work, Ops&&... ops) {
	S slv(params, std::forward<W>(work));
	return std::move(slv).bind(std::forward<Ops>(ops)...);
}


namespace op {
template <class W, template <class> class Slv>
struct factory<krylov_interface<W, Slv>> {
	template<class P>
	static auto create(P&& params) {
		return std::apply([&](auto&&...v) {
			if constexpr (std::is_rvalue_reference_v<P&&>) {
				return make_krylov_op<Slv>(params.solver_settings,
				                           std::forward<decltype(params.work)>(params.work),
				                           std::forward<decltype(v)>(v)...);
			} else {
				return make_krylov_op<Slv>(params.solver_settings,
				                           params.work,
				                           v...);
			}
		}, std::forward<decltype(params.ops)>(params.ops));
	}
};
}

}
#endif
