#ifndef FLECSOLVE_SOLVERS_KRYLOV_PARAMETERS_H
#define FLECSOLVE_SOLVERS_KRYLOV_PARAMETERS_H

#include <tuple>
#include <memory>

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
	krylov_parameters_gen(O &&... o) : ops(std::forward<O>(o)...) {}

	template<class T>
	static bool default_diagnostic(const T &, double) {
		return false;
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

	std::shared_ptr<solver_type> solver;

protected:
	std::tuple<std::decay_t<Ops>...> ops;
};

template<bool precond_is_factory, class solver_type, class... Ops>
struct krylov_parameters_base {};

template<class solver_type, class... Ops>
struct krylov_parameters_base<false, solver_type, Ops...>
	: krylov_parameters_gen<solver_type, Ops...> {
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
		     (I > 1 || is_operator_v<std::tuple_element_t<I, decltype(ops)>>)));
	}
};

template<class solver_type, class... Ops>
struct krylov_parameters_base<true, solver_type, Ops...>
	: with_label, krylov_parameters_gen<solver_type, Ops...> {

	using base_t = krylov_parameters_gen<solver_type, Ops...>;
	using base_t::ops;
	using base_t::solver;

	template<class... O>
	krylov_parameters_base(std::string pre,
	                       std::function<void(void)> callback,
	                       O &&... o)
		: with_label(pre.c_str()), base_t(std::forward<O>(o)...),
		  create_solver(callback) {
		// assert correct number of operators are provided
		static_assert(sizeof...(O) >= 2 &&
		              sizeof...(O) <= krylov_oplabel::nops);
		// assert A is an operator
		static_assert(is_operator_v<std::tuple_element_t<0, decltype(ops)>>);
	}

	template<krylov_oplabel lb>
	decltype(auto) get_operator() {
		return base_t::template get_operator<lb>();
	}

	template<>
	decltype(auto) get_operator<krylov_oplabel::P>() {
		auto & factory = std::get<krylov_oplabel::P>(ops);
		if (!factory.has_solver()) {
			factory.create(
				get_workvec(*solver),
				std::ref(
					base_t::template get_operator_ref<krylov_oplabel::A>()));
		}
		return op::shell([&factory, this](auto & x, auto & y) {
			return factory.solve(
				x,
				y,
				get_workvec(*solver),
				std::ref(
					base_t::template get_operator_ref<krylov_oplabel::A>()));
		});
	}

	auto options() {
		namespace po = boost::program_options;
		po::options_description desc;
		auto & factory = std::get<krylov_oplabel::P>(ops);
		desc.add_options()(label("preconditioner").c_str(),
		                   po::value<std::string>()->required()->notifier(
							   [&factory](const std::string & name) {
								   factory.set_options_name(name);
							   }),
		                   "preconditioner name");
		desc.add(factory.options([&factory,
		                          this](const typename std::decay_t<
										decltype(factory)>::registry & reg) {
			if (!solver)
				create_solver();
			factory.create_parameters(
				reg,
				get_workvec(*solver),
				std::ref(
					base_t::template get_operator_ref<krylov_oplabel::A>()));
		}));

		return desc;
	}

protected:
	template<class Work>
	auto & get_workvec(krylov_interface<Work> & kint) {
		return std::get<0>(kint.work);
	}
	std::function<void(void)> create_solver;
};

namespace detail {
template<class... Ops>
struct precond_is_factory : std::false_type {};

template<class O, class P, class... Rest>
struct precond_is_factory<O, P, Rest...> {
	static constexpr bool value = !is_operator_v<P>;
};

template<class... Ops>
static constexpr bool precond_is_factory_v = precond_is_factory<Ops...>::value;
}

template<class SP, class SW, class... Ops>
struct krylov_parameters
	: krylov_parameters_base<detail::precond_is_factory_v<Ops...>,
                             typename flecsolve::traits<
								 std::decay_t<SP>>::template solver_type<SW>,
                             Ops...> {
	using solver_type =
		typename flecsolve::traits<std::decay_t<SP>>::template solver_type<SW>;
	static constexpr bool precond_is_factory =
		detail::precond_is_factory_v<Ops...>;
	using base_t =
		krylov_parameters_base<precond_is_factory, solver_type, Ops...>;
	using base_t::solver;

	template<class P,
	         class W,
	         class... O,
	         std::enable_if_t<detail::precond_is_factory_v<O...>, bool> = true>
	krylov_parameters(P && sp, W && sw, O &&... ops)
		: base_t(sp.get_prefix(),
	             std::bind(&krylov_parameters::create_solver, this),
	             std::forward<O>(ops)...),
		  solver_settings(std::forward<P>(sp)),
		  solver_work(std::forward<W>(sw)) {}

	template<class P,
	         class W,
	         class... O,
	         std::enable_if_t<!detail::precond_is_factory_v<O...>, bool> = true>
	krylov_parameters(P && sp, W && sw, O &&... ops)
		: base_t(std::forward<O>(ops)...), solver_settings(std::forward<P>(sp)),
		  solver_work(std::forward<W>(sw)) {}

	auto & get_solver() {
		if (not solver) {
			create_solver();
		}
		return *solver;
	}

	auto options() {
		auto desc = solver_settings.options();
		desc.add(base_t::options());
		return desc;
	}

	std::decay_t<SP> solver_settings;
	std::decay_t<SW> solver_work;

protected:
	void create_solver() {
		solver = std::make_shared<solver_type>(
			std::forward<SP>(solver_settings), std::forward<SW>(solver_work));
	}
};
template<class SP, class SW, class... Ops>
krylov_parameters(SP &&, SW &&, Ops &&...) -> krylov_parameters<SP, SW, Ops...>;

}
#endif
