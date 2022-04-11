#pragma once

#include <tuple>
#include <utility>

namespace flecsi {
namespace linalg {
namespace operators {

#include <cassert>
#include <functional>
#include <tuple>

template<class T>
struct OpExpr;

template<class... Ps>
struct OpExpr<std::tuple<Ps...>>
{
  std::tuple<Ps...> ops;

  constexpr OpExpr(Ps... ps) : ops(std::forward<Ps>(ps)...) {}

  template<class U, class V>
  constexpr void apply(U&& u, V&& v) const
  {
    std::apply(
        [&](auto&&... a) {
          (a.apply(std::forward<decltype(u)>(u), std::forward<decltype(v)>(v)),
           ...);
        },
        ops);
  }

  constexpr decltype(auto) get_parameters() const
  {
    return _gp(std::make_index_sequence<sizeof...(Ps)> {});
  }

  template<std::size_t... II>
  constexpr decltype(auto) _gp(std::index_sequence<II...>) const
  {
    return std::make_tuple(
        typename std::decay_t<
            std::tuple_element_t<II, decltype(ops)>>::param_type {}...);
  }
  // reset
};

template<class... Ps>
inline constexpr auto op_expr(Ps&&... ps)
{
  return OpExpr<std::tuple<Ps...>> {std::forward<Ps>(ps)...};
}

}  // namespace operators
}  // namespace linalg
}  // namespace flecsi