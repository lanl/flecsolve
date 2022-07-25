#pragma once

#include <cstddef>

namespace heat_eqn {

using scalar_t = double;

constexpr std::size_t NX = 8;
constexpr std::size_t NY = 8;
constexpr std::size_t NZ = 1;

enum class heateqn_var { v1};

constexpr scalar_t DEFAULT_VAL = 1.0;

} // namespace heat_eqn