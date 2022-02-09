#pragma once

#include <string>
#include <array>
#include <type_traits>

namespace flecsi::linalg {

template <auto V> struct variable_name {
  static constexpr const char *value = "";
};

template <auto V> struct variable_t {
	static constexpr auto val = V;
	static constexpr const char * name = variable_name<V>::value;
};

template<auto V>
inline variable_t<V> variable{};

}
