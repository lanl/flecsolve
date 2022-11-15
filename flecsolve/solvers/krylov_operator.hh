#ifndef FLECSOLVE_SOLVERS_KRYLOV_OPERATOR_H
#define FLECSOLVE_SOLVERS_KRYLOV_OPERATOR_H

#include "flecsolve/operators/base.hh"
#include "flecsolve/operators/traits.hh"
#include "flecsolve/solvers/traits.hh"
#include "flecsolve/solvers/krylov_parameters.hh"

namespace flecsolve {
namespace op {

template<class Params>
struct krylov : op::base<krylov<Params>> {
	using base_t = op::base<krylov<Params>>;
	using base_t::input_var;
	using base_t::output_var;
	using base_t::params;

	krylov(Params p) : op::base<krylov<Params>>(std::move(p)) {}

	template<class D, class R>
	auto apply(const vec::base<D> & b, vec::base<R> & x) {
		decltype(auto) bs = b.subset(input_var);
		decltype(auto) xs = x.subset(output_var);

		flog_assert(xs != bs, "Input and output vectors must be distinct");

		auto & solver = params.get_solver();
		decltype(auto) diag =
			params.template get_operator<krylov_oplabel::diag>();
		decltype(auto) precond =
			params.template get_operator<krylov_oplabel::P>();
		auto & op = params.template get_operator<krylov_oplabel::A>();

		return solver.apply(op, bs, xs, precond, diag);
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
private:
	template<class T>
	struct op_vars {
		static constexpr auto input_var = traits<T>::input_var;
		static constexpr auto output_var = traits<T>::output_var;
	};

	template<class T>
	struct op_vars<std::reference_wrapper<T>> {
		static constexpr auto input_var = op_vars<T>::input_var;
		static constexpr auto output_var = op_vars<T>::output_var;
	};

public:
	using parameters = P;
	// krylov operators inherit variables from the operator they invert.
	using vars =
		op_vars<std::tuple_element_t<0, decltype(std::declval<P>().ops)>>;
	static constexpr auto input_var = vars::input_var;
	static constexpr auto output_var = vars::output_var;
};

template<class Params, class... Ops>
auto rebind(krylov<Params> & kr, Ops &&... ops) {
	static_assert(!detail::precond_is_factory_v<Ops...>);
	krylov_parameters_base<false, typename Params::solver_type, Ops...>
		new_params(
			"", []() {}, std::forward<Ops>(ops)...);
	new_params.solver = kr.params.solver;
	return krylov(std::move(new_params));
}

}
}
#endif
