#ifndef FLECSI_LINALG_VECTORS_VARIABLE_H
#define FLECSI_LINALG_VECTORS_VARIABLE_H

#include <string>
#include <array>
#include <type_traits>
#include <limits>

namespace flecsi::linalg {

enum class anon_var : std::size_t { anonymous = std::numeric_limits<std::size_t>::max() };

template <auto V> struct variable_name {
  static constexpr const char *value = "";
};

template <auto V> struct variable_t {
	static constexpr auto val = V;
	static constexpr const char * name = variable_name<V>::value;
};

template<auto V>
inline variable_t<V> variable{};

template <auto... Vs> struct varlist {};

}
#endif
