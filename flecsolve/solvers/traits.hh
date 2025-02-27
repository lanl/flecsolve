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
