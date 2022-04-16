#pragma once

#include <cstddef>

namespace diffusion {

using scalar_t = double;

constexpr std::size_t NX = 8;
constexpr std::size_t NY = 8;

enum class diffusion_var { v1, v2 };

constexpr scalar_t DEFAULT_VAL = 1.0;

constexpr auto diff_alpha = scalar_t{0.0};
constexpr auto diff_beta = DEFAULT_VAL;

} // namespace diffusion