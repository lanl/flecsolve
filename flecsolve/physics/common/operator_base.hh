#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <flecsi/util/array_ref.hh>
#include <flecsi/util/constant.hh>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace flecsolve {
namespace physics {

	/**
	* !!THIS DESIGN IS SUBJECT TO CHANGE!!
	*
	* `operators` are constructed through specilization of 3 required and 1 optional classes:
	*
	*	operator_traits<specalized>: defines compile-time features of the operator; holds no run-time data.
	*	operator_parameters<specalized>: defines the parameters of the operator, i.e. all run-time data
	*	tasks::operator_task<specalized>: the run-time execution, should define `launch()`
	*	(optional) operator_creator<specialized>: holds the particular operator construction.
	*
	*/

template<class Derived>
struct operator_traits;
template<class Derived>
struct operator_parameters;
template<class Derived>
struct operator_creator;

namespace tasks {
template<class Derived>
struct operator_task;
}

template<class Derived, template<class> class BaseType>
struct operator_base {
	using exact_type = Derived;

	constexpr exact_type const & derived() const {
		return static_cast<exact_type const &>(*this);
	}

	constexpr exact_type & derived() {
		return static_cast<exact_type &>(*this);
	}

	template<class U, class V>
	constexpr decltype(auto) apply(U && u, V && v) const {
		return derived().apply(std::forward<U>(u), std::forward<V>(v));
	}

private:
	operator_base() {}
	friend BaseType<Derived>;
};

template<class Derived, template<class> class BaseType>
struct operator_traits<operator_base<Derived, BaseType>> {
	static constexpr std::string_view label{"base"};
};

template<class Derived>
struct operator_settings : operator_base<Derived, operator_settings> {

	using base_type = operator_base<Derived, operator_settings>;
	using exact_type = typename base_type::exact_type;
	using param_type = operator_parameters<Derived>;
	using task_type = tasks::operator_task<Derived>;

	template<class P>
	operator_settings(P && p) : base_type() {
		this->reset(p);
	}

	template<class... Ps>
	constexpr static param_type getParamType(Ps &&... ps) {
		return param_type{ps...};
	}

	template<auto CPH, class VPH>
	constexpr decltype(auto) get_parameters(VPH &) const {
		return parameters;
	}

	template<class P>
	constexpr decltype(auto) reset(P && pars) {
		parameters = pars;
	}

	constexpr decltype(auto) flat() const { return std::make_tuple(*this); }

	const auto to_string() const { return operator_traits<exact_type>::label; }

	template<class... Args>
	static constexpr auto create(param_type && pars, Args &&... args) {
		return exact_type(args..., std::forward<param_type>(pars));
	}

	// static constexpr auto get_label()
	// {
	// 	return std::string_view{std::string(base_type::get_label()) +
	// std::string("::") + std::string(exact_type::label)};
	// }
	// private:

	param_type parameters;
	friend exact_type;
};

template<typename>
struct is_tuple : std::false_type {};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

}
}
