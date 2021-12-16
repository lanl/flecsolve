#pragma once

#include <flecsi/data.hh>

namespace flecsi::linalg::vec::data {

static constexpr flecsi::partition_privilege_t na = flecsi::na;
static constexpr flecsi::partition_privilege_t ro = flecsi::ro;
static constexpr flecsi::partition_privilege_t wo = flecsi::wo;
static constexpr flecsi::partition_privilege_t rw = flecsi::rw;

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

	static inline constexpr flecsi::PrivilegeCount num_priv =
		topo_t::template privilege_count<Space>;
	static inline constexpr flecsi::Privileges read_only_dofs =
		flecsi::privilege_cat<flecsi::privilege_repeat<ro, num_priv - (num_priv > 1)>,
		                      flecsi::privilege_repeat<na, (num_priv > 1)>>;
	static inline constexpr flecsi::Privileges write_only_dofs =
		flecsi::privilege_cat<flecsi::privilege_repeat<wo, num_priv - (num_priv > 1)>,
		                      flecsi::privilege_repeat<na, (num_priv > 1)>>;
	static inline constexpr flecsi::Privileges read_write_dofs =
		flecsi::privilege_cat<flecsi::privilege_repeat<rw, num_priv - (num_priv > 1)>,
		                      flecsi::privilege_repeat<na, (num_priv > 1)>>;
	static inline constexpr flecsi::Privileges read_only_all =
		flecsi::privilege_repeat<ro, num_priv>;
	static inline constexpr flecsi::Privileges write_only_all =
		flecsi::privilege_repeat<wo, num_priv>;

	using topo_acc = typename topo_t::template accessor<ro>;
	using ro_acc = typename field<real_t>::template accessor1<read_only_dofs>;
	using wo_acc = typename field<real_t>::template accessor1<write_only_dofs>;
	using rw_acc = typename field<real_t>::template accessor1<read_write_dofs>;

	using ro_acc_all = typename field<real_t>::template accessor1<read_only_all>;
	using wo_acc_all = typename field<real_t>::template accessor1<write_only_all>;

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
