#ifndef FLECSOLVE_SOLVERS_KRYLOV_INTERFACE_H
#define FLECSOLVE_SOLVERS_KRYLOV_INTERFACE_H

#include <tuple>
#include <memory>

#include "flecsolve/util/traits.hh"
#include "flecsolve/operators/traits.hh"
#include "flecsolve/operators/shell.hh"
#include "flecsolve/solvers/traits.hh"

namespace flecsolve {
namespace op {

enum krylov_oplabel : std::size_t { A, P, diag, nops };

template<class solver_type, class... Ops>
struct krylov_parameters_base {

	template<class... O>
	krylov_parameters_base(O &&... o) : ops(std::forward<O>(o)...) {
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

template<class Params>
struct krylov : op::base<krylov<Params>> {
	using op::base<krylov<Params>>::params;

	krylov(Params p) : op::base<krylov<Params>>(std::move(p)) {}

	template<class D, class R>
	auto apply(const vec::base<D> & b, vec::base<R> & x) {
		auto & op = params.template get_operator<krylov_oplabel::A>();
		decltype(auto) bs = subset_input(b, op);
		decltype(auto) xs = subset_output(x, op);

		flog_assert(xs != bs, "Input and output vectors must be distinct");

		auto & solver = params.get_solver();
		decltype(auto) diag =
			params.template get_operator<krylov_oplabel::diag>();
		auto & precond = params.template get_operator<krylov_oplabel::P>();

		return solver.apply(op, bs, xs, precond, diag);
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

	template<class T>
	void reset(const T & settings) {
		auto & solver = params.get_solver();
		solver.reset(settings);
	}

	auto & get_operator() {
		return params.template get_operator<krylov_oplabel::A>();
	}

	const auto & get_operator() const {
		return params.template get_operator<krylov_oplabel::A>();
	}
};
template<class P>
krylov(P) -> krylov<P>;

template<class P>
struct traits<krylov<P>> {
	using parameters = P;
	static constexpr auto input_var = variable<anon_var::anonymous>;
	static constexpr auto output_var = variable<anon_var::anonymous>;
};

template<class Params, class... Ops>
auto rebind(krylov<Params> & kr, Ops &&... ops) {
	krylov_parameters_base<typename Params::solver_type, Ops...> new_params(
		std::forward<Ops>(ops)...);
	new_params.solver = kr.params.solver;
	return krylov(std::move(new_params));
}

}

template<class Workspace, template<class> class Solver>
struct krylov_interface {
	using workvec_t = typename std::remove_reference_t<Workspace>::value_type;
	using real = typename workvec_t::real;

	Workspace work;
};

}
#endif
