#ifndef FLECSI_LINALG_VECTORS_TOPO_VIEW_HH
#define FLECSI_LINALG_VECTORS_TOPO_VIEW_HH

#include "base.hh"
#include "variable.hh"
#include "operations/topo_view.hh"
#include "data/topo_view.hh"

namespace flecsolve::vec {

template<auto V, class Topo, typename Topo::index_space Space, class Scalar>
struct topo_view : base<topo_view<V, Topo, Space, Scalar>> {
	using base_t = base<topo_view<V, Topo, Space, Scalar>>;

	template<class Slot, class Ref>
	topo_view(variable_t<V>, Slot & topo, Ref ref)
		: base_t(data::topo_view<Topo, Space, Scalar>{topo, ref}) {}

	template<auto var>
	const auto & subset(variable_t<var>) const {
		static_assert(var == V);
		return *this;
	}

	template<auto var>
	auto & subset(variable_t<var>) {
		static_assert(var == V);
		return *this;
	}
};

template<auto V, class Slot, class Ref>
topo_view(variable_t<V>, Slot &, Ref) -> topo_view<V,
                                                   typename Ref::Topology,
                                                   Ref::space,
                                                   typename Ref::value_type>;

template<class Topo, typename Topo::index_space Space, class Scalar>
struct topo_view<anon_var::anonymous, Topo, Space, Scalar>
	: base<topo_view<anon_var::anonymous, Topo, Space, Scalar>> {
	using base_t = base<topo_view<anon_var::anonymous, Topo, Space, Scalar>>;

	template<class Slot, class Ref>
	topo_view(Slot & topo, Ref ref)
		: base_t(data::topo_view<Topo, Space, Scalar>{std::ref(topo), ref}) {}

	template<auto var>
	const auto & subset(variable_t<var>) const {
		static_assert(var == anon_var::anonymous);
		return *this;
	}

	template<auto var>
	auto & subset(variable_t<var>) {
		static_assert(var == anon_var::anonymous);
		return *this;
	}
};

template<class Slot, class Ref>
topo_view(Slot &, Ref) -> topo_view<anon_var::anonymous,
                                    typename Ref::Topology,
                                    Ref::space,
                                    typename Ref::value_type>;
}

namespace flecsolve {
template<auto V, class Topo, typename Topo::index_space Space, class Scalar>
struct traits<vec::topo_view<V, Topo, Space, Scalar>> {
	static constexpr auto var = variable<V>;
	using data_t = vec::data::topo_view<Topo, Space, Scalar>;
	using ops_t = vec::ops::topo_view<Topo, Space, Scalar>;
};
}
#endif
