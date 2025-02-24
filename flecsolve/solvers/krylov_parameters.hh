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

#include "flecsolve/operators/storage.hh"
#include "flecsolve/operators/shell.hh"
#include "flecsolve/util/traits.hh"
#include "flecsolve/util/config.hh"

namespace flecsolve {

template<class Workspace>
struct krylov_interface {
	using workvec_t = typename std::remove_reference_t<Workspace>::value_type;
	using real = typename workvec_t::real;

	Workspace work;
};

}

namespace flecsolve::op {

enum krylov_oplabel : std::size_t { A, P, diag, nops };

template<class solver_type, class... Ops>
struct krylov_parameters_gen {

	template<class... O>
	krylov_parameters_gen(O &&... o) : ops(storage{std::forward<O>(o)}...) {}

	template<class T>
	static bool default_diagnostic(const T &, double) {
		return false;
	}

	auto & get_solver() { return *solver; }

	template<krylov_oplabel lb>
	auto & get_operator_ref() {
		return std::get<lb>(ops).get();
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

	template<krylov_oplabel lb>
	void set_operator(
		std::tuple_element_t<lb, std::tuple<storage<std::decay_t<Ops>>...>>
			new_op) {
		std::get<lb>(ops) = new_op;
	}

	std::shared_ptr<solver_type> solver;

protected:
	std::tuple<storage<std::decay_t<Ops>>...> ops;
};

template<class solver_type, class Derived, class... Ops>
struct krylov_parameters_base : krylov_parameters_gen<solver_type, Ops...> {
	using base_t = krylov_parameters_gen<solver_type, Ops...>;
	using base_t::ops;

	template<class... O>
	krylov_parameters_base(O &&... o) : base_t(std::forward<O>(o)...) {
		// assert correct number of operators are provided
		static_assert(sizeof...(O) >= 1 &&
		              sizeof...(O) <= krylov_oplabel::nops);
		// assert both operator and preconditioner inherit from op::base
		assert_operators(std::make_index_sequence<sizeof...(Ops)>());
	}

	auto options() {
		namespace po = boost::program_options;
		po::options_description desc;
		return desc;
	}

protected:
	template<std::size_t... I>
	void assert_operators(std::index_sequence<I...>) {
		// operator and preconditioner types must be operators
		static_assert(
			(... &&
		     (I > 1 ||
		      is_operator_v<
				  typename std::tuple_element_t<I, decltype(ops)>::op_type>)));
	}
};

template<class SP, class SW, class... Ops>
struct krylov_parameters
	: krylov_parameters_base<typename flecsolve::traits<
								 std::decay_t<SP>>::template solver_type<SW>,
                             krylov_parameters<SP, SW, Ops...>,
                             Ops...> {
	using solver_type =
		typename flecsolve::traits<std::decay_t<SP>>::template solver_type<SW>;
	using base_t = krylov_parameters_base<solver_type,
	                                      krylov_parameters<SP, SW, Ops...>,
	                                      Ops...>;
	using base_t::solver;

	template<class P, class W, class... O>
	krylov_parameters(P && sp, W && sw, O &&... ops)
		: base_t(std::forward<O>(ops)...), solver_settings(std::forward<P>(sp)),
		  solver_work(std::forward<W>(sw)) {}

	auto & get_solver() {
		create_solver();
		return *solver;
	}

	auto options() {
		auto desc = solver_settings.options();
		desc.add(base_t::options());
		return desc;
	}

	std::decay_t<SP> solver_settings;
	std::decay_t<SW> solver_work;

	void create_solver() {
		if (!solver) {
			solver = std::make_shared<solver_type>(
				solver_settings, std::forward<SW>(solver_work));
		}
	}
};
template<class SP, class SW, class... Ops>
krylov_parameters(SP &&, SW &&, Ops &&...) -> krylov_parameters<SP, SW, Ops...>;

}
#endif
