#ifndef FLECSI_LINALG_VEC_DATA_TOPO_VIEW_HH
#define FLECSI_LINALG_VEC_DATA_TOPO_VIEW_HH

#include "flecsolve/topo/csr.hh"
#include <flecsi/data.hh>
#include <flecsi/topo/core.hh>
#include <flecsi/topo/narray/interface.hh>
#include <flecsi/topo/unstructured/interface.hh>
#include <utility>

namespace flecsolve::vec::data {

/*
 * select the topology accessor with the correct
 * permissions
 *
 * we use `base` alias type defined in `flecsi::specialization`
 * to check if the mesh shares the same core topo type.
 *
 * NOTE: flecsi itself doesn't have a great way to do this,
 * so this is here until the that lib provides this.
 */
// the default case; this handles narray, csr.
// these could be specialized explicitly but this
// was less verbose
template<class T, class = void>
struct topo_acc_select {
	using acc = typename T::template accessor<flecsi::ro>;
};
// the case of unstructured, with three permission bits
template<class t>
struct topo_acc_select<t,
                       typename std::enable_if_t<std::is_same_v<
						   typename t::base,
						   flecsi::topo::base_t<flecsi::topo::unstructured>>>> {
	using acc =
		typename t::template accessor<flecsi::ro, flecsi::ro, flecsi::ro>;
};

template<typename T, flecsi::data::layout L = flecsi::data::layout::dense>
using field = flecsi::field<T, L>;

template<class Config>
struct topo_view {
	using config = Config;
	using scalar = typename Config::scalar;
	using topo_t = typename Config::topo_t;
	static constexpr typename topo_t::index_space space = Config::space;
	using topo_slot_t = flecsi::data::topology_slot<topo_t>;
	using field_definition =
		typename field<scalar>::template definition<topo_t, space>;
	using field_reference =
		typename field<scalar>::template Reference<topo_t, space>;

	static inline constexpr flecsi::PrivilegeCount num_priv =
		topo_t::template privilege_count<space>;

	template<flecsi::partition_privilege_t priv>
	static inline constexpr flecsi::Privileges dofs_priv =
		flecsi::privilege_cat<
			flecsi::privilege_repeat<priv, num_priv - (num_priv > 1)>,
			flecsi::privilege_repeat<flecsi::na, (num_priv > 1)>>;

	// select the topology accessor witt the correct number of
	// permissions flags
	using topo_acc = typename topo_acc_select<topo_t>::acc;

	template<flecsi::partition_privilege_t priv>
	using acc = typename field<scalar>::template accessor1<dofs_priv<priv>>;

	template<flecsi::partition_privilege_t priv>
	using acc_all = typename field<scalar>::template accessor1<
		flecsi::privilege_repeat<priv, num_priv>>;

	struct util {
		template<class U>
		static decltype(auto) dofs(U && m) {
			return std::forward<U>(m).template dofs<space>();
		}
	};

	std::reference_wrapper<topo_slot_t> topo_slot;
	field_reference reference;

	auto ref() const { return reference; }
	auto fid() const { return ref().fid(); }
	topo_slot_t & topo() const { return topo_slot; }
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
