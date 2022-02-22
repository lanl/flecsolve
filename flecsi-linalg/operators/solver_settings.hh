#pragma once

#include <array>

namespace flecsi::linalg {

template<class Op, class Vec, std::size_t NumWork>
struct solver_settings {
	static constexpr std::size_t num_work = NumWork;
	using real = typename Vec::real;
	using vec = Vec;

	int maxiter;
	real rtol;
	real atol;
	Op precond;
	std::array<Vec, num_work> work;
};


template <class Vec, std::size_t NumWork, std::size_t Version=0>
struct topo_solver_state {

	using field_def = typename Vec::data_t::field_definition;
	using topo_slot_t = typename Vec::data_t::topo_slot_t;
	static inline std::array<const field_def, NumWork> defs;

	static auto get_work(Vec & rhs) {
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


}
