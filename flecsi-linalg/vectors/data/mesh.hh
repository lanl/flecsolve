#ifndef FLECSI_LINALG_VEC_DATA_MESH_H
#define FLECSI_LINALG_VEC_DATA_MESH_H

#include <flecsi/data.hh>

namespace flecsolve::vec::data {

template<typename T, flecsi::data::layout L = flecsi::data::layout::dense>
using field = flecsi::field<T, L>;

template<class Topo, typename Topo::index_space Space, class T>
struct mesh {
	using topo_t = Topo;
	static constexpr typename Topo::index_space space = Space;
	using topo_slot_t = flecsi::data::topology_slot<Topo>;
	using field_definition =
		typename field<T>::template definition<Topo, Space>;
	using field_reference = typename field<T>::template Reference<Topo, Space>;

	static inline constexpr flecsi::PrivilegeCount num_priv =
		topo_t::template privilege_count<Space>;

	template<flecsi::partition_privilege_t priv>
	static inline constexpr flecsi::Privileges dofs_priv =
		flecsi::privilege_cat<
			flecsi::privilege_repeat<priv, num_priv - (num_priv > 1)>,
			flecsi::privilege_repeat<flecsi::na, (num_priv > 1)>>;

	using topo_acc = typename topo_t::template accessor<flecsi::ro>;
	template<flecsi::partition_privilege_t priv>
	using acc = typename field<T>::template accessor1<dofs_priv<priv>>;

	template<flecsi::partition_privilege_t priv>
	using acc_all = typename field<T>::template accessor1<
		flecsi::privilege_repeat<priv, num_priv>>;

	struct util {
		template<class U>
		static decltype(auto) dofs(U && m) {
			return std::forward<U>(m).template dofs<space>();
		}
	};

	topo_slot_t & topo;
	field_reference reference;

	auto ref() const { return reference; }
	auto fid() const { return ref().fid(); }
};

template<class Slot, class Ref>
mesh(Slot &, Ref)
	-> mesh<typename Ref::Base::Topology, Ref::space, typename Ref::value_type>;

template<class Topo, typename Topo::index_space Space, class T>
bool operator==(const mesh<Topo, Space, T> & d1,
                const mesh<Topo, Space, T> & d2) {
	return d1.fid() == d2.fid();
}

template<class Topo, typename Topo::index_space Space, class T>
bool operator!=(const mesh<Topo, Space, T> & d1,
                const mesh<Topo, Space, T> & d2) {
	return d1.fid() != d2.fid();
}

}

#endif
