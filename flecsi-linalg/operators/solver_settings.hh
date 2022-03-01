#pragma once

#include <array>

namespace flecsi::linalg {

template<class Op, class Diag>
struct solver_settings {
	using diagnostic_t = Diag;

	int maxiter;
	float rtol;
	float atol;
	Op & precond;
	Diag diagnostic;
};
template <class Op, class Diag>
solver_settings(int,float,float,Op&,Diag&&)->solver_settings<Op,Diag>;


struct solve_stats {};

struct solve_info {
	solve_info() : iters(0) {}

	enum class stop_reason {
		converged_atol, converged_rtol, converged_user,
		diverged_dtol, diverged_iters, diverged_breakdown
	};
	stop_reason status;
	int iters;
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


template <class Settings, class Workspace, template<class,class> class Solver>
struct solver_interface {
	using workvec_t = typename std::remove_reference_t<Workspace>::value_type;
	using real = typename workvec_t::real;

	template<class ... Args>
	constexpr bool user_diagnostic(Args&&... args) {
		if constexpr (!std::is_null_pointer_v<typename Settings::diagnostic_t>) {
			return settings.diagnostic(std::forward<Args>(args)...);
		} else
			return false;
	}

	Settings settings;
	Workspace work;
};

}
