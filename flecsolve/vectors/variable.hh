#ifndef FLECSI_LINALG_VECTORS_VARIABLE_H
#define FLECSI_LINALG_VECTORS_VARIABLE_H

#include <string>
#include <array>
#include <type_traits>
#include <limits>

namespace flecsolve {

enum class anon_var : std::size_t {
	anonymous = std::numeric_limits<std::size_t>::max()
};

template<auto V>
struct variable_name {
	static constexpr const char * value = "";
};

template<auto V>
struct variable_t {
	static constexpr auto value = V;
	static constexpr const char * name = variable_name<V>::value;
};
template<auto V0, auto V1>
constexpr bool operator==(const variable_t<V0> &, const variable_t<V1> &) {
	return V0 == V1;
}
template<auto V0, auto V1>
constexpr bool operator!=(const variable_t<V0> &, const variable_t<V1> &) {
	return V0 != V1;
}

template<auto V>
inline variable_t<V> variable{};

template<class T>
struct is_variable : std::false_type {};

template<auto V>
struct is_variable<variable_t<V>> : std::true_type {};

template<class T>
inline constexpr bool is_variable_v = is_variable<T>::value;

template<auto... Vs>
struct multivariable_t {};

template<auto... Vs>
inline multivariable_t<Vs...> multivariable{};

template<auto... V0, auto... V1>
constexpr bool operator==(const multivariable_t<V0...> &,
                          const multivariable_t<V1...> &) {
	return (... && (V0 == V1));
}

template<auto... V0, auto... V1>
constexpr bool operator!=(const multivariable_t<V0...> &,
                          const multivariable_t<V1...> &) {
	return (... || (V0 != V1));
}

}
#endif
