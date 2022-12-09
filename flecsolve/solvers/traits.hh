#ifndef FLECSI_LINALG_OP_TRAITS_H
#define FLECSI_LINALG_OP_TRAITS_H

#include <type_traits>
#include <functional>

#include "flecsolve/solvers/solver_settings.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve::op {

template<class T, class V, typename = void>
struct is_solver : std::false_type {};

template<class T, class V>
struct is_solver<T,
                 V,
                 typename std::enable_if<std::is_same_v<
					 decltype(std::declval<T>().apply(std::declval<V &>(),
                                                      std::declval<V &>())),
					 solve_info>>::type> : std::true_type {};

template<class T, class V>
struct is_solver<std::reference_wrapper<T>,
                 V,
                 typename std::enable_if<std::is_same_v<
					 decltype(std::declval<T>().apply(std::declval<V &>(),
                                                      std::declval<V &>())),
					 solve_info>>::type> : std::true_type {};

template<class T, class V>
inline constexpr bool is_solver_v = is_solver<T, V>::value;

}
#endif
