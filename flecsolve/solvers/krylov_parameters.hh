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
#ifndef FLECSOLVE_SOLVERS_KRYLOV_PARAMETERS_H
#define FLECSOLVE_SOLVERS_KRYLOV_PARAMETERS_H

#include <tuple>
#include <memory>

#include "flecsolve/operators/handle.hh"
#include "flecsolve/operators/shell.hh"
#include "flecsolve/util/traits.hh"
#include "flecsolve/util/config.hh"

namespace flecsolve {

static inline const auto default_diagnostic = [](auto &&, double) {
	return false;
};

template<class Op, class Precond, class Diag>
struct krylov_ops
{
	op::handle<Op> A;
	op::handle<Precond> P;
	std::decay_t<Diag> diagnostic;
};

template<class Settings, class Work, class Op, class... Rest>
struct krylov_parameters {
	using work_t = Work;
	using workvec_t = typename std::remove_reference_t<Work>::value_type;
	using real = typename workvec_t::real;
	using scalar = typename workvec_t::scalar;
	using op_type = Op;
	using input_var_t = decltype(op_type::input_var);
	using output_var_t = decltype(op_type::output_var);

	template<class S, class W, class A, class P, class D>
	krylov_parameters(S && s, W && w,
	                  op::handle<A> a, op::handle<P> p,
	                  D && d) :
		work{std::forward<W>(w)},
		settings{std::forward<S>(s)},
		ops{a, p, std::forward<D>(d)} {}
	template<class S, class W, class A, class P>
	krylov_parameters(S && s, W && w,
	                  op::handle<A> a, op::handle<P> p) :
		krylov_parameters(std::forward<S>(s),
		                  std::forward<W>(w), a, p,
		                  default_diagnostic) {}

	template<class S, class W, class A>
	krylov_parameters(S && s, W && w,
	                  op::handle<A> a) :
		krylov_parameters(std::forward<S>(s),
		                  std::forward<W>(w),
		                  a, op::make_identity(A::input_var, A::output_var)) {}

	const auto & A() const { return ops.A.get(); }
	const auto & P() const { return ops.P.get(); }

	mutable Work work;
	Settings settings;
	krylov_ops<Op, Rest...> ops;
};

template<class S, class W, class A>
krylov_parameters(S&&, W&&, op::handle<A>) ->
	krylov_parameters<std::decay_t<S>, std::decay_t<W>, A,
	                  typename decltype(op::make_identity(A::input_var, A::output_var))::type,
	                  decltype(default_diagnostic)>;

template<class S, class W, class A, class P>
krylov_parameters(S &&, W &&, op::handle<A>, op::handle<P>)
	-> krylov_parameters<std::decay_t<S>, std::decay_t<W>, A, P, decltype(default_diagnostic)>;

template<class S, class W, class A, class P, class D>
krylov_parameters(S&&, W&&, op::handle<A>, op::handle<P>, D&&) ->
	krylov_parameters<std::decay_t<S>, std::decay_t<W>, A, P, D>;

template<template<class> class OpType, class Settings, class Workspace>
struct krylov_solver {
	using settings_type = Settings;

	template<class ... Ops>
	auto bind(Ops && ... ops) && {
		return OpType(
			krylov_parameters(
				std::move(settings), std::move(workspace),
				std::forward<Ops>(ops)...));
	}

	template<class ... Ops>
	auto bind(Ops && ... ops) & {
		return OpType(
			krylov_parameters(
				settings, workspace,
				std::forward<Ops>(ops)...));
	}

	template<class ... Ops>
	auto operator()(Ops && ... ops) && {
		return op::make(std::move(*this).bind(std::forward<Ops>(ops)...));
	}

	template<class ... Ops>
	auto operator()(Ops && ... ops) & {
		return op::make(bind(std::forward<Ops>(ops)...));
	}

	settings_type settings;
	Workspace workspace;
};


}
#endif
