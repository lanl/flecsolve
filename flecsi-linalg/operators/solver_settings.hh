#pragma once

#include <array>

#include "shell.hh"

namespace flecsi::linalg {

struct solver_settings {
	int maxiter;
	float rtol;
	float atol;
};


struct solve_stats {};

struct solve_info {
	solve_info() : iters(0), restarts(0) {}

	enum class stop_reason {
		converged_atol, converged_rtol, converged_user,
		diverged_dtol, diverged_iters, diverged_breakdown
	};
	stop_reason status;
	int iters;
	int restarts;
	float res_norm_initial, res_norm_final;
	float sol_norm_initial, sol_norm_final;
	float rhs_norm;
};


template <class Vec, std::size_t NumWork, std::size_t Version>
struct topo_solver_state {

	using field_def = typename Vec::data_t::field_definition;
	using topo_slot_t = typename Vec::data_t::topo_slot_t;
	static inline std::array<const field_def, NumWork> defs;

	static auto get_work(const Vec & rhs) {
		return make_work(rhs.data.topo, defs, std::make_index_sequence<NumWork>());
	}

protected:
	template<std::size_t ... Index>
	static std::array<Vec, NumWork> make_work(
		topo_slot_t & slot, std::array<const field_def,
		NumWork> & defs,
		std::index_sequence<Index...>) {
		return { Vec(slot, defs[Index](slot))... };
	}
};


template<std::size_t NumWork, std::size_t Version>
struct topo_work_base {
	template<class Vec>
	static auto get(const Vec & rhs) {
		return topo_solver_state<Vec, NumWork, Version>::get_work(rhs);
	}
};


template <class S, class P, class D>
struct bound_op {
	S & slv;
	P & pre;
	D diag;

	template<class Op, class DomainVec, class RangeVec>
	auto apply(const Op & A, const RangeVec & b, DomainVec & x) {
		return slv.apply(A, b, x, pre, diag);
	}
};
template <class S, class P, class D>
bound_op(S&,P&,D&&)->bound_op<S,P,D>;

template <class Settings, class Workspace, template<class> class Solver>
struct solver_interface {
	using workvec_t = typename std::remove_reference_t<Workspace>::value_type;
	using real = typename workvec_t::real;

	template<class Op, class DomainVec, class RangeVec>
	auto apply(const Op & A, const RangeVec & b, DomainVec & x) {
		return static_cast<
			Solver<Workspace>&>(*this).apply(A, b, x, op::I, [](auto...){ return false; });
	}

	template<class Op, class DomainVec, class RangeVec, class Precond>
	auto apply(const Op & A, const RangeVec & b, DomainVec & x, Precond & pre) {
		return static_cast<
			Solver<Workspace>&>(*this).apply(A, b, x, pre, [](auto...){ return false; });
	}

	template<class P, class D>
	auto bind(P & pre, D && diag) {
		return bound_op{static_cast<Solver<Workspace>&>(*this), pre, diag};
	}

	Settings settings;
	Workspace work;
};

}
