#pragma once

#include <array>

namespace flecsi::linalg {

template<class Op>
struct solver_settings {
	int maxiter;
	float rtol;
	float atol;
	Op precond;
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

	template<class Op, class DomainVec, class RangeVec>
	void apply(const Op & A, const RangeVec & b, DomainVec & x)
	{
		static_cast<Solver<Settings,Workspace>&>(*this).apply(A, b, x, nullptr);
	}

	template<class F, class ... Args>
	static constexpr void invoke(F && f, Args&&... args) {
		if constexpr (!std::is_null_pointer_v<F>) {
			std::forward<F>(f)(std::forward<Args>(args)...);
		}
	}

	Settings settings;
	Workspace work;
};

}
