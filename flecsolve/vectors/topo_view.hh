#ifndef FLECSI_LINALG_VECTORS_TOPO_VIEW_HH
#define FLECSI_LINALG_VECTORS_TOPO_VIEW_HH

#include "core.hh"
#include "variable.hh"
#include "operations/topo_view.hh"
#include "data/topo_view.hh"

namespace flecsolve::vec {

template<auto V, class Scalar, class Topo, typename Topo::index_space Space>
struct topo_view_config {
	using scalar = Scalar;
	using real = typename num_traits<scalar>::real;
	using len_t = flecsi::util::id;
	static constexpr auto var = variable<V>;
	using var_t = decltype(V);
	static constexpr std::size_t num_components = 1;
	using topo_t = Topo;
	static constexpr typename Topo::index_space space = Space;
};

template<auto V, class Scalar, class Topo, typename Topo::index_space Space>
struct topo_view : core<data::topo_view,
                        ops::topo_view,
                        topo_view_config<V, Scalar, Topo, Space>> {

	using config_t = topo_view_config<V, Scalar, Topo, Space>;
	using base = core<data::topo_view, ops::topo_view, config_t>;

	template<class Slot, class Ref>
	topo_view(variable_t<V>, Slot & topo, Ref ref)
		: base(data::topo_view<config_t>{topo, ref}) {}

	template<class Slot, class Ref>
	topo_view(Slot & topo, Ref ref)
		: base(data::topo_view<config_t>{topo, ref}) {}
};

template<auto V, class Slot, class Ref>
topo_view(variable_t<V>, Slot &, Ref) -> topo_view<V,
                                                   typename Ref::value_type,
                                                   typename Ref::Topology,
                                                   Ref::space>;
template<class Slot, class Ref>
topo_view(Slot &, Ref) -> topo_view<anon_var::anonymous,
                                    typename Ref::value_type,
                                    typename Ref::Topology,
                                    Ref::space>;

}
#endif
