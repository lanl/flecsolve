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
