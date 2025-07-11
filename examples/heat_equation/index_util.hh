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
#ifndef FLECSOLVE_EXAMPLES_HEAT_INDEX_UTIL_H
#define FLECSOLVE_EXAMPLES_HEAT_INDEX_UTIL_H
#include <flecsi/data.hh>

namespace heat::util {

inline static std::size_t digit(flecsi::util::id & x, std::size_t d) {
	std::size_t ret = x % d;
	x /= d;
	return ret;
}

struct srange {
	std::size_t beg, end;
	std::size_t size() const { return end - beg; }
};

inline flecsi::util::id translate(flecsi::util::id & x,
                                  std::size_t & stride,
                                  const srange & sub,
                                  std::size_t extent) {
	flecsi::util::id ret = digit(x, sub.size()) + sub.beg;

	ret *= stride;
	stride *= extent;

	return ret;
}

template<std::size_t... Index>
flecsi::util::id
translate_index(flecsi::util::id x,
                const std::array<srange, sizeof...(Index)> & subrange,
                const std::array<std::size_t, sizeof...(Index)> & extents,
                std::index_sequence<Index...>) {
	std::size_t stride = 1;
	return (translate(x, stride, subrange[Index], extents[Index]) + ...);
}

template<std::size_t... Index>
std::size_t subrange_size(std::array<srange, sizeof...(Index)> & subrange,
                          std::index_sequence<Index...>) {
	return (subrange[Index].size() * ...);
}

template<auto S, unsigned short Dim>
inline auto make_subrange_ids(std::array<srange, Dim> subrange,
                              std::array<std::size_t, Dim> extents) {
	return flecsi::util::transform_view(
		flecsi::util::iota_view<flecsi::util::id>(
			0, subrange_size(subrange, std::make_index_sequence<Dim>())),
		[=](const auto & x) {
			return flecsi::topo::id<S>(translate_index(
				x, subrange, extents, std::make_index_sequence<Dim>()));
		});
}

}
#endif
