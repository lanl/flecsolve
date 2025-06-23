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
#ifndef FLECSI_LINALG_VEC_DATA_TOPO_VIEW_HH
#define FLECSI_LINALG_VEC_DATA_TOPO_VIEW_HH

#include <flecsi/data.hh>

namespace flecsolve::vec::data {

template<class Config>
struct topo_view {
	using config = Config;
	using scalar = typename Config::scalar;
	using topo_t = typename Config::topo_t;
	static constexpr typename topo_t::index_space space = Config::space;

	template<class T = typename Config::scalar>
	using field = flecsi::field<scalar, Config::layout>;

	using field_definition =
		typename flecsi::field<scalar, Config::layout>::template
		definition<topo_t, space>;
	using field_reference =
		typename flecsi::field<scalar, Config::layout>::template Reference<topo_t, space>;

	static inline constexpr flecsi::PrivilegeCount num_priv =
		topo_t::template privilege_count<space>;

	template<flecsi::privilege priv>
	static inline constexpr flecsi::Privileges dofs_priv =
		flecsi::privilege_cat<
			flecsi::privilege_repeat<priv, num_priv - (num_priv > 1)>,
			flecsi::privilege_repeat<flecsi::na, (num_priv > 1)>>;

	using topo_acc = flecsi::data::topology_accessor<
		topo_t,
		flecsi::privilege_repeat<flecsi::ro, num_priv>>;

	template<flecsi::privilege priv>
	using acc = typename flecsi::field<scalar, Config::layout>::template accessor1<dofs_priv<priv>>;

	template<flecsi::privilege priv>
	using acc_all = typename flecsi::field<scalar, Config::layout>::template accessor1<
		flecsi::privilege_repeat<priv, num_priv>>;

	struct util {
		template<class U>
		static decltype(auto) dofs(U && m) {
			return std::forward<U>(m).template dofs<space>();
		}
	};

	field_reference reference;

	auto ref() const { return reference; }
	auto fid() const { return ref().fid(); }
	auto & topo() const { return ref().topology(); }
};

template<class Config>
bool operator==(const topo_view<Config> & d1, const topo_view<Config> & d2) {
	return d1.fid() == d2.fid();
}

template<class Config>
bool operator!=(const topo_view<Config> & d1, const topo_view<Config> & d2) {
	return d1.fid() != d2.fid();
}

}

#endif
