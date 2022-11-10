#ifndef FLECSOLVE_SOLVERS_KRYLOV_PARAMETERS_H
#define FLECSOLVE_SOLVERS_KRYLOV_PARAMETERS_H

#include <tuple>
#include <memory>

#include "flecsolve/operators/shell.hh"
#include "flecsolve/util/traits.hh"

namespace flecsolve::op {

enum krylov_oplabel : std::size_t { A, P, diag, nops };

template<class solver_type, class... Ops>
struct krylov_parameters_base {

	template<class... O>
	krylov_parameters_base(O &&... o) : ops(std::forward<O>(o)...) {
		// assert correct number of operators are provided
		static_assert(sizeof...(O) >= 1 &&
		              sizeof...(O) <= krylov_oplabel::nops);
		// assert both operator and preconditioner inherit from op::base
		assert_operators(std::make_index_sequence<sizeof...(Ops)>());
	}

	auto & get_solver() { return *solver; }

	template<krylov_oplabel lb>
	auto & get_operator_ref() {
		if constexpr (is_reference_wrapper_v<
						  std::tuple_element_t<lb, decltype(ops)>>)
			return std::get<lb>(ops).get();
		else
			return std::get<lb>(ops);
	}

	template<krylov_oplabel lb>
	decltype(auto) get_operator() {
		if constexpr (lb > 0 && lb >= sizeof...(Ops)) {
			// get default
			if constexpr (lb == krylov_oplabel::P)
				return op::I;
			else if constexpr (lb == krylov_oplabel::diag) {
				return [](const auto &, double) { return false; };
			}
		}
		else {
			return get_operator_ref<lb>();
		}
	}

	template<class T>
	static bool default_diagnostic(const T &, double) {
		return false;
	}

	std::shared_ptr<solver_type> solver;
	std::tuple<std::decay_t<Ops>...> ops;

protected:
	template<std::size_t... I>
	void assert_operators(std::index_sequence<I...>) {
		// operator and preconditioner types must be operators
		static_assert(
			(... &&
		     (I > 1 || is_operator_v<std::tuple_element_t<I, decltype(ops)>>)));
	}
};

template<class SP, class SW, class... Ops>
struct krylov_parameters
	: krylov_parameters_base<typename flecsolve::traits<
								 std::decay_t<SP>>::template solver_type<SW>,
                             Ops...> {
	using solver_type =
		typename flecsolve::traits<std::decay_t<SP>>::template solver_type<SW>;
	using base_t = krylov_parameters_base<solver_type, Ops...>;
	using base_t::solver;

	template<class P, class W, class... O>
	krylov_parameters(P && sp, W && sw, O &&... ops)
		: base_t(std::forward<O>(ops)...), solver_settings(std::forward<P>(sp)),
		  solver_work(std::forward<W>(sw)) {}

	auto & get_solver() {
		if (not solver) {
			solver =
				std::make_shared<solver_type>(std::forward<SP>(solver_settings),
			                                  std::forward<SW>(solver_work));
		}
		return *solver;
	}

	auto options() { return solver_settings.options(); }

	std::decay_t<SP> solver_settings;
	std::decay_t<SW> solver_work;
};
template<class SP, class SW, class... Ops>
krylov_parameters(SP &&, SW &&, Ops &&...) -> krylov_parameters<SP, SW, Ops...>;

}
#endif
