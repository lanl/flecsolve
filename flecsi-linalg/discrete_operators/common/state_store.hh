#pragma once

#include <array>
#include <cstddef>
#include <flecsi/execution.hh>
#include <flecsi/util/array_ref.hh>
#include <flecsi/util/constant.hh>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace flecsi {
namespace linalg {
namespace operators {
namespace common {

template<class K, auto Space, std::size_t Len, std::size_t tag>
struct topo_state_store
{
  using topo_t = typename K::topo_t;
  using scalar_t = typename K::scalar_t;
  using topo_slot_t = typename K::topo_slot_t;

  using fd = typename field<scalar_t>::template definition<topo_t, Space>;
  using fd_ref = typename field<scalar_t>::template Reference<topo_t, Space>;

  static inline std::array<fd, Len> fields;

  static auto get_state(topo_slot_t& s)
  {
    return make_state(s, fields, std::make_index_sequence<Len>());
  }

 protected:
  template<std::size_t... Index>
  static decltype(auto) make_state(topo_slot_t& slot,
                                   std::array<fd, Len>& f,
                                   std::index_sequence<Index...>)
  {
    if constexpr (Len == 1) {
      return f[0](slot);
    } else {
      return std::array<fd_ref, Len> {f[Index](slot)...};
    }
  }
};

}  // namespace common
}  // namespace operators
}  // namespace linalg
}  // namespace flecsi