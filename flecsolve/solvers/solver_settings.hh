#ifndef FLECSI_LINALG_OP_SOLVER_SETTINGS_H
#define FLECSI_LINALG_OP_SOLVER_SETTINGS_H

#include <array>
#include <type_traits>

#include <boost/program_options/options_description.hpp>

#include "flecsolve/vectors/multi.hh"

namespace flecsolve {

namespace po = boost::program_options;

struct solver_settings {
	solver_settings(const char * pre) : prefix(pre) {}

	auto options() {
		po::options_description desc;
		// clang-format off
		desc.add_options()
			(label("maxiter").c_str(), po::value<int>(&maxiter)->required(), "maximum number of iterations")
			(label("rtol").c_str(), po::value<float>(&rtol)->default_value(0), "relative tolerance")
			(label("atol").c_str(), po::value<float>(&atol)->default_value(0), "absolute tolerance")
			(label("use-zero-guess").c_str(), po::value<bool>(&use_zero_guess)->required(), "use zero inital guess");
		// clang-format on
		return desc;
	}

	int maxiter;
	float rtol;
	float atol;
	bool use_zero_guess;

protected:
	std::string prefix;
	std::string label(const char * suf) { return {prefix + "." + suf}; }
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

	bool success() {
		return (status == stop_reason::converged_atol) ||
		       (status == stop_reason::converged_rtol) ||
		       (status == stop_reason::converged_user);
	}
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
			rhs.data.topo(), defs, std::make_index_sequence<NumWork>());
	}

protected:
	template<std::size_t... Index>
	static std::array<Vec, NumWork>
	make_work(topo_slot_t & slot,
	          std::array<const field_def, NumWork> & defs,
	          std::index_sequence<Index...>) {
		if constexpr (std::is_same_v<typename Vec::var_t, anon_var>)
			return {Vec(slot, defs[Index](slot))...};
		else
			return {Vec(Vec::var, slot, defs[Index](slot))...};
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
		return make(std::move(wv), std::make_index_sequence<NumWork>());
	}

protected:
	template<class T, std::size_t... Index>
	static auto make_states(T & t, std::index_sequence<Index...>) {
		return std::make_tuple(
			topo_solver_state<std::remove_cv_t<std::remove_reference_t<
								  std::tuple_element_t<Index, T>>>,
		                      NumWork,
		                      Version,
		                      Index>::get_work(std::get<Index>(t))...);
	}

	template<class T, std::size_t Index>
	static auto make_mv(T && wv) {
		return std::apply(
			[](auto &&... v) { return vec::multi(std::move(v[Index])...); },
			std::forward<T>(wv));
	}

	template<class T, std::size_t... Index>
	static auto make(T && wv, std::index_sequence<Index...>) {
		return std::array{make_mv<T, Index>(std::forward<T>(wv))...};
	}
};

}
#endif
