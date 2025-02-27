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
#ifndef FLECSOLVE_VECTORS_TRAITS_HH
#define FLECSOLVE_VECTORS_TRAITS_HH

#include "core.hh"

namespace flecsolve {

template<class T>
class is_vector
{
	template<template<class> class D, template<class> class O, class C>
	static decltype(static_cast<vec::core<D, O, C>>(std::declval<T>()),
	                std::true_type{}) test(const vec::core<D, O, C> &);
	static std::false_type test(...);

public:
	static constexpr bool value =
		decltype(is_vector::test(std::declval<T>()))::value;
};

template<class T>
static constexpr bool is_vector_v = is_vector<T>::value;

}
#endif
