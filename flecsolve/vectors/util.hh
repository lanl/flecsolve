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
#ifndef FLECSI_LINALG_VECTORS_UTIL_HH
#define FLECSI_LINALG_VECTORS_UTIL_HH

#include <tuple>
#include <istream>

namespace flecsolve::vec {

template<class F, class... Vecs>
constexpr decltype(auto) apply(F && f, Vecs &&... vecs) {
	using fvec =
		std::tuple_element_t<0, std::tuple<std::remove_reference_t<Vecs>...>>;
	static_assert((... && (fvec::num_components ==
	                       std::remove_reference_t<Vecs>::num_components)));
	if constexpr (fvec::num_components == 1)
		return fvec::ops::apply(std::forward<F>(f),
		                        std::forward<Vecs>(vecs)...);
	else
		return fvec::ops::apply(std::forward<F>(f),
		                        std::forward<Vecs>(vecs).data...);
}

enum class norm_type { inf, l2, l1 };

std::istream & operator>>(std::istream &, norm_type &);

}

#endif
