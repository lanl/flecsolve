/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
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

#include "flecsolve/physics/common/vector_types.hh"

namespace flecsolve {
namespace physics {
namespace common {

template<class K, class Vec, auto Space, std::size_t Len, std::size_t tag>
struct topo_state_store {
	using topo_t = physics::topo_t<Vec>;
	using scalar_t = physics::scalar_t<Vec>;

	using fd = field_def<Vec, Space>;
	// ypename flecsi::field<scalar_t>::template definition<topo_t, Space>;
	using fd_ref = field_ref<Vec, Space>;
	// typename flecsi::field<scalar_t>::template Reference<topo_t, Space>;

	static inline std::array<fd, Len> fields;

	static auto get_state(typename topo_t::topology & s) {
		return make_state(s, fields, std::make_index_sequence<Len>());
	}

protected:
	template<std::size_t... Index>
	static decltype(auto) make_state(typename topo_t::topology & slot,
	                                 std::array<fd, Len> & f,
	                                 std::index_sequence<Index...>) {
		if constexpr (Len == 1) {
			return f[0](slot);
		}
		else {
			return std::array<fd_ref, Len>{f[Index](slot)...};
		}
	}
};

}
}
}
