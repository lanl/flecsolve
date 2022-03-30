#ifndef FLECSI_LINALG_UTIL_TRAITS_H
#define FLECSI_LINALG_UTIL_TRAITS_H

#include <complex>

namespace flecsi::linalg {

template <class T> struct traits {};

template <class T>
struct num_traits {
	using scalar = T;
	using real = T;
	static constexpr bool is_complex = false;
};

template <class T>
struct num_traits<std::complex<T>>
{
	using scalar = std::complex<T>;
	using real = T;
	static constexpr bool is_complex = true;
};

}
#endif
