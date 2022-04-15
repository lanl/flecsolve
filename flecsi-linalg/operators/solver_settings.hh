#ifndef FLECSI_LINALG_OP_SOLVER_SETTINGS_H
#define FLECSI_LINALG_OP_SOLVER_SETTINGS_H

#include "flecsi-linalg/vectors/multi.hh"
#include "shell.hh"
#include <array>
#include <type_traits>

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
		converged_atol,
		converged_rtol,
		converged_user,
		diverged_dtol,
		diverged_iters,
		diverged_breakdown
	};
	stop_reason status;
	int iters;
	int restarts;
	float res_norm_initial, res_norm_final;
	float sol_norm_initial, sol_norm_final;
	float rhs_norm;
};

template<class Vec,
         std::size_t NumWork,
         std::size_t Version,
         std::size_t MVIndex = 0>
struct topo_solver_state {

	using field_def = typename Vec::data_t::field_definition;
	using topo_slot_t = typename Vec::data_t::topo_slot_t;
	static inline std::array<const field_def, NumWork> defs = {};

	static auto get_work(const Vec & rhs) {
		return make_work(
			rhs.data.topo, defs, std::make_index_sequence<NumWork>());
	}

protected:
	template<std::size_t... Index>
	static std::array<Vec, NumWork>
	make_work(topo_slot_t & slot,
	          std::array<const field_def, NumWork> & defs,
	          std::index_sequence<Index...>) {
		if constexpr (std::is_same_v<std::decay_t<decltype(Vec::var)>,
		                             decltype(anon_var::anonymous)>)
			return {Vec(slot, defs[Index](slot))...};
		else
			return {Vec(variable<Vec::var>, slot, defs[Index](slot))...};
	}
};

template<std::size_t NumWork, std::size_t Version>
struct topo_work_base {
	template<class Vec>
	static auto get(const Vec & rhs) {
		return topo_solver_state<Vec, NumWork, Version>::get_work(rhs);
	}

	template<class VarType, class... Vecs>
	static auto get(const vec::multi<VarType, Vecs...> & rhs) {
		auto wv =
			make_states(rhs.data, std::make_index_sequence<sizeof...(Vecs)>());
		return make(rhs, wv, std::make_index_sequence<NumWork>());
	}

protected:
	template<class T, std::size_t... Index>
	static auto make_states(T & t, std::index_sequence<Index...>) {
		return std::make_tuple(
			topo_solver_state<
				std::remove_reference_t<std::tuple_element_t<Index, T>>,
				NumWork,
				Version,
				Index>::get_work(std::get<Index>(t))...);
	}

	template<class T, class MV, std::size_t Index>
	static MV make_mv(const T & wv) {
		return std::apply([](const auto &... v) { return MV(v[Index]...); },
		                  wv);
	}

	template<class MV, class T, std::size_t... Index>
	static std::array<MV, NumWork>
	make(const MV &, const T & wv, std::index_sequence<Index...>) {
		return {make_mv<T, MV, Index>(wv)...};
	}
};

} // namespace flecsi::linalg
#endif
