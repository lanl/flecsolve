#ifndef FLECSOLVE_SOLVERS_KRYLOV_OPERATOR_H
#define FLECSOLVE_SOLVERS_KRYLOV_OPERATOR_H

#include "flecsolve/operators/core.hh"
#include "flecsolve/operators/traits.hh"
#include "flecsolve/solvers/traits.hh"
#include "flecsolve/solvers/krylov_parameters.hh"

namespace flecsolve {
namespace op {

namespace detail {
template<class P>
struct config {
private:
	template<class T>
	struct op_vars {
		static constexpr auto input_var = T::input_var;
		static constexpr auto output_var = T::output_var;
	};

public:
	// krylov operators inherit variables from the operator they invert.
	using vars = op_vars<typename std::tuple_element_t<
		0,
		decltype(std::declval<P>().ops)>::op_type>;
	static constexpr auto input_var = vars::input_var;
	static constexpr auto output_var = vars::output_var;
};
template<class P>
using krylov_base =
	base<P, decltype(config<P>::input_var), decltype(config<P>::output_var)>;
}

template<class Params>
struct krylov : detail::krylov_base<Params> {
	using base_t = detail::krylov_base<Params>;
	using base_t::input_var;
	using base_t::output_var;
	using base_t::params;

	krylov(Params p) : base_t(std::move(p)) {}

	template<class D, class R>
	auto apply(const D & b, R & x) {
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

	template<class D, class R>
	auto apply(const D & b, R & x) const {
		return const_cast<krylov<Params> &>(*this).apply(b, x);
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

template<class P, template<class> class S = shared_storage>
auto krylov_solver(P && p) {
	return core<krylov<P>, S>(std::forward<P>(p));
}

template<class Params, class... Ops>
auto rebind(krylov<Params> & kr, Ops &&... ops) {
	static_assert(!detail::precond_is_factory_v<Ops...>);
	krylov_parameters_base<false,
	                       typename Params::solver_type,
	                       std::nullptr_t,
	                       Ops...>
	new_params(std::forward<Ops>(ops)...);
	new_params.solver = kr.params.solver;
	return krylov(std::move(new_params));
}

}
}
#endif
