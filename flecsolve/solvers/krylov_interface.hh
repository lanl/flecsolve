#ifndef FLECSI_LINALG_OP_KRYLOV_INTERFACE_H
#define FLECSI_LINALG_OP_KRYLOV_INTERFACE_H

#include <tuple>

#include "traits.hh"
#include "shell.hh"
#include "factory.hh"

namespace flecsolve {

template<class S, class A, class P, class D>
struct krylov_op {
	S solver;
	A op;
	P precond;
	D diag;

	template<class DomainVec, class RangeVec>
	auto apply(const RangeVec & b, DomainVec & x) {
		decltype(auto) bs = subset_input(b, op);
		decltype(auto) xs = subset_output(x, op);

		flog_assert(xs != bs, "Input and output vectors must be distinct");

		return solver.apply(op, bs, xs, precond, diag);
	}

	A & get_operator() { return op; }
	const A & get_oeprator() const { return op; }

	template<class T>
	void reset(const T & settings) {
		solver.reset(settings);
	}

	template<class... Args>
	auto rebind(Args &&... args) & {
		return solver.bind(std::forward<Args>(args)...);
	}
	template<class... Args>
	auto rebind(Args &&... args) && {
		return std::forward<S>(solver).bind(std::forward<Args>(args)...);
	}

	template<class T, class O>
	static decltype(auto) subset_input(const T & x, const O &) {
		static_assert(op::has_input_variable_v<O>);
		return x.subset(O::input_var);
	}

	template<class T, class O>
	static decltype(auto) subset_output(T & x, const O &) {
		static_assert(op::has_output_variable_v<O>);
		return x.subset(O::output_var);
	}

	static constexpr auto input_var = variable<anon_var::anonymous>;
	static constexpr auto output_var = variable<anon_var::anonymous>;
};
template<class S, class A, class P, class D>
krylov_op(S &&, A &&, P &&, D &&) -> krylov_op<S, A, P, D>;

template<class Workspace, template<class> class Solver>
struct krylov_interface {
	using workvec_t = typename std::remove_reference_t<Workspace>::value_type;
	using real = typename workvec_t::real;

	template<class A>
	auto bind(A && op) & {
		return krylov_op{static_cast<Solver<Workspace> &>(*this),
		                 std::forward<A>(op),
		                 op::I,
		                 [](const auto &, double) { return false; }};
	}

	template<class A>
	auto bind(A && op) && {
		return krylov_op{std::move(*static_cast<Solver<Workspace> *>(this)),
		                 std::forward<A>(op),
		                 op::I,
		                 [](const auto &, double) { return false; }};
	}

	template<class A, class P>
	auto bind(A && op, P && p) & {
		return krylov_op{static_cast<Solver<Workspace> &>(*this),
		                 std::forward<A>(op),
		                 std::forward<P>(p),
		                 [](const auto &, double) { return false; }};
	}

	template<class A, class P>
	auto bind(A && op, P && p) && {
		return krylov_op{std::move(*static_cast<Solver<Workspace> *>(this)),
		                 std::forward<A>(op),
		                 std::forward<P>(p),
		                 [](const auto &, double) { return false; }};
	}

	template<class A, class P, class D>
	auto bind(A && op, P && p, D && d) & {
		return krylov_op{static_cast<Solver<Workspace> &>(*this),
		                 std::forward<A>(op),
		                 std::forward<P>(p),
		                 std::forward<D>(d)};
	}

	template<class A, class P, class D>
	auto bind(A && op, P && p, D && d) && {
		return krylov_op{std::move(*static_cast<Solver<Workspace> *>(this)),
		                 std::forward<A>(op),
		                 std::forward<P>(p),
		                 std::forward<D>(d)};
	}

	Workspace work;
};

template<class S, class W, class... Ops>
struct krylov_params {
	using settings_type = S;

	template<class V, class... O>
	krylov_params(settings_type s, V && w, O &&... o)
		: solver_settings(std::move(s)), work(std::forward<V>(w)),
		  ops(std::forward<O>(o)...) {}

	settings_type solver_settings;
	W work;
	std::tuple<Ops...> ops;
};
template<class S, class W, class... O>
krylov_params(S, W &&, O &&...) -> krylov_params<S, W, O...>;

template<template<class> class S, class W, class... Ops>
auto make_krylov_op(const typename S<W>::settings_type & params,
                    W && work,
                    Ops &&... ops) {
	S slv(params, std::forward<W>(work));
	return std::move(slv).bind(std::forward<Ops>(ops)...);
}

namespace op {
template<class W, template<class> class Slv>
struct factory<krylov_interface<W, Slv>> {
	template<class P>
	static auto create(P && params) {
		return std::apply(
			[&](auto &&... v) {
				if constexpr (std::is_rvalue_reference_v<P &&>) {
					return make_krylov_op<Slv>(
						params.solver_settings,
						std::forward<decltype(params.work)>(params.work),
						std::forward<decltype(v)>(v)...);
				}
				else {
					return make_krylov_op<Slv>(
						params.solver_settings, params.work, v...);
				}
			},
			std::forward<decltype(params.ops)>(params.ops));
	}
};
}

}
#endif
