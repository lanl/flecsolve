#pragma once

#include <flecsi/data.hh>

namespace flecsi::linalg::vec::data {

template<typename T, flecsi::data::layout L = flecsi::data::layout::dense>
using field = flecsi::field<T, L>;

template<class Topo, typename Topo::index_space Space, class Real=double>
struct mesh {
	using real_t = Real;
	using len_t = std::size_t;
	using topo_t = Topo;
	static constexpr typename Topo::index_space space = Space;
	using topo_slot_t = flecsi::data::topology_slot<Topo>;
	using field_definition = typename flecsi::field<real_t>::template definition<Topo, Space>;

	static inline constexpr PrivilegeCount num_priv =
		topo_t::template privilege_count<Space>;

	template<partition_privilege_t priv>
	static inline constexpr Privileges dofs_priv =
		privilege_cat<privilege_repeat<priv, num_priv - (num_priv > 1)>,
		              privilege_repeat<na, (num_priv > 1)>>;

	using topo_acc = typename topo_t::template accessor<ro>;
	template<partition_privilege_t priv>
	using acc = typename field<real_t>::template accessor1<dofs_priv<priv>>;

	template<partition_privilege_t priv>
	using acc_all = typename field<real_t>::template accessor1<privilege_repeat<priv, num_priv>>;

	struct util
	{
		template<class T>
		static decltype(auto) dofs(T&& m) {
			return std::forward<T>(m).template dofs<space>();
		}
	};

	const field_definition & def;
	topo_slot_t & topo;

	auto ref() const { return def(topo); }
	auto fid() const { return ref().fid(); }
};

}
