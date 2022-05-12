#ifndef FLECSI_LINALG_OP_TRAITS_H
#define FLECSI_LINALG_OP_TRAITS_H

#include "flecsolve/vectors/variable.hh"

namespace flecsolve::op {

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
