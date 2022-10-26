#ifndef FLECSOLVE_OP_TRAITS_H
#define FLECSOLVE_OP_TRAITS_H

#include <type_traits>
#include <functional>

#include "flecsolve/vectors/base.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve::op {

enum class label { jacobian };

template<class T>
struct traits {
	static constexpr auto input_var = variable<anon_var::anonymous>;
	static constexpr auto output_var = variable<anon_var::anonymous>;
};

template<class T, class = void>
struct has_input_variable_t : std::false_type {};
template<class T>
struct has_input_variable_t<T, decltype((void)T::input_var, void())>
	: std::true_type {};
template<class T>
inline constexpr bool has_input_variable_v = has_input_variable_t<T>::value;

template<class T, class = void>
struct has_output_variable_t : std::false_type {};
template<class T>
struct has_output_variable_t<T, decltype((void)T::output_var, void())>
	: std::true_type {};
template<class T>
inline constexpr bool has_output_variable_v = has_output_variable_t<T>::value;

}

#endif
