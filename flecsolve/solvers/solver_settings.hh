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
#ifndef FLECSI_LINALG_OP_SOLVER_SETTINGS_H
#define FLECSI_LINALG_OP_SOLVER_SETTINGS_H

#include <array>
#include <type_traits>

#include <boost/program_options/options_description.hpp>

#include "flecsolve/vectors/topo_view.hh"
#include "flecsolve/vectors/multi.hh"

namespace flecsolve {

namespace po = boost::program_options;

struct solver_settings {
	int maxiter;
	float rtol;
	float atol;
	bool use_zero_guess;
};
struct solver_options {
	using settings_type = solver_settings;
	solver_options(const char * pre) : prefix(pre) {}

	auto operator()(settings_type & settings) {
		po::options_description desc;
		// clang-format off
		desc.add_options()
			(label("maxiter").c_str(), po::value<int>(&settings.maxiter)->required(), "maximum number of iterations")
			(label("rtol").c_str(), po::value<float>(&settings.rtol)->default_value(0), "relative tolerance")
			(label("atol").c_str(), po::value<float>(&settings.atol)->default_value(0), "absolute tolerance")
			(label("use-zero-guess").c_str(), po::value<bool>(&settings.use_zero_guess)->required(), "use zero inital guess");
		// clang-format on
		return desc;
	}
	const std::string & get_prefix() const { return prefix; }

protected:
	std::string prefix;
	std::string label(const char * suf) { return {prefix + "." + suf}; }
};

struct solve_stats {};

struct solve_info {
	solve_info() : iters(0), restarts(0), status(stop_reason::unknown) {}

	enum class stop_reason {
		converged_atol,
		converged_rtol,
		converged_user,
		diverged_dtol,
		diverged_iters,
		diverged_breakdown,
		unknown
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

inline std::ostream & operator<<(std::ostream & os, const solve_info::stop_reason & r) {
	switch (r) {
	case solve_info::stop_reason::converged_atol:
		os << "Converged to absolute tolerance";
		break;
	case solve_info::stop_reason::converged_rtol:
		os << "Converged to residual tolerance";
		break;
	case solve_info::stop_reason::converged_user:
		os << "Converged to user tolerance";
		break;
	case solve_info::stop_reason::diverged_iters:
		os << "Diverged: exceeded maximum iterations";
		break;
	case solve_info::stop_reason::diverged_breakdown:
		os << "Diverged due to breakdown";
		break;
	case solve_info::stop_reason::diverged_dtol:
		os << "Diverged: reached divergence tolerance";
		break;
	case solve_info::stop_reason::unknown:
		os << "Status unknown";
		break;
	};

	return os;
}

template<std::size_t version>
struct version_t {
	static constexpr std::size_t value = version;
};

template<std::size_t V>
inline version_t<V> version;

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
			return {vec::make(slot, defs[Index](slot))...};
		else
			return {vec::make(Vec::var, slot, defs[Index](slot))...};
	}
};

template<std::size_t NumWork, std::size_t Version>
struct topo_work_base {
	template<class Vec>
	static auto get(const Vec & rhs) {
		return topo_solver_state<Vec, NumWork, Version>::get_work(rhs);
	}

	template<class... Vecs>
	static auto get(const vec::multi<Vecs...> & rhs) {
		auto wv = make_states(rhs.data.components,
		                      std::make_index_sequence<sizeof...(Vecs)>());
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

template<std::size_t nwork>
struct work_factory {
	template<class Vec>
	constexpr auto operator()(Vec & b) const {
		return topo_work_base<nwork, 0>::get(b);
	}

	template<class Vec, std::size_t Ver>
	constexpr auto operator()(Vec & b, version_t<Ver>) const {
		return topo_work_base<nwork, Ver>::get(b);
	}
};
}
#endif
