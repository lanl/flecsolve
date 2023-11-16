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

// https://stackoverflow.com/a/47906253
namespace detail {
template<typename Struct, typename = void, typename... T>
struct is_direct_list_initializable_impl : std::false_type {};

template<typename Struct, typename... T>
struct is_direct_list_initializable_impl<
	Struct,
	std::void_t<decltype(Struct{std::declval<T>()...})>,
	T...> : std::true_type {};
}

template<typename Struct, typename... T>
using is_direct_list_initializable =
	detail::is_direct_list_initializable_impl<Struct, void, T...>;

template<typename Struct, typename... T>
constexpr bool is_direct_list_initializable_v =
	is_direct_list_initializable<Struct, T...>::value;
template<typename Struct, typename... T>
using is_aggregate_initializable = std::conjunction<
	std::is_aggregate<Struct>,
	is_direct_list_initializable<Struct, T...>,
	std::negation<std::conjunction<
		std::bool_constant<sizeof...(T) == 1>,
		std::is_same<std::decay_t<std::tuple_element_t<0, std::tuple<T...>>>,
                     Struct>>>>;

template<typename Struct, typename... T>
constexpr bool is_aggregate_initializable_v =
	is_aggregate_initializable<Struct, T...>::value;

}
#endif
