#pragma once

#include <cstddef>

namespace eqdiff {

using scalar_t = double;

constexpr std::size_t NX = 4;
constexpr std::size_t NY = 4;
constexpr std::size_t NZ = 1;

constexpr std::size_t NVAR = 2;
enum class diffusion_var { v1, v2 };

constexpr scalar_t DEFAULT_VAL = 1.0;

} // namespace diffusion