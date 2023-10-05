#ifndef FLECSI_LINALG_UTIL_TRAITS_H
#define FLECSI_LINALG_UTIL_TRAITS_H

#include <complex>
#include <memory>
#include <utility>
#include <functional>
#include <type_traits>

namespace flecsolve {

template<class T>
struct is_reference_wrapper : std::false_type {};
template<class T>
struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type {};

template<class T>
inline constexpr bool is_reference_wrapper_v = is_reference_wrapper<T>::value;

template<class T>
struct is_smart_ptr : std::false_type {};

template<class T, class D>
struct is_smart_ptr<std::unique_ptr<T, D>> : std::true_type {};

template<class T>
struct is_smart_ptr<std::shared_ptr<T>> : std::true_type {};

template<class T>
inline constexpr bool is_smart_ptr_v = is_smart_ptr<T>::value;

template<class T>
struct traits {};

template<class T>
struct num_traits {
	using scalar = T;
	using real = T;
	static constexpr bool is_complex = false;
};

template<class T>
struct num_traits<std::complex<T>> {
	using scalar = std::complex<T>;
	using real = T;
	static constexpr bool is_complex = true;
};

template<class Derived>
struct with_derived {
	Derived & derived() { return static_cast<Derived &>(*this); }

	const Derived & derived() const {
		return static_cast<const Derived &>(*this);
	}
};

}
#endif
