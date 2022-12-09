#ifndef FLECSI_LINALG_VECTORS_MESH_H
#define FLECSI_LINALG_VECTORS_MESH_H

#include "base.hh"
#include "variable.hh"
#include "operations/mesh.hh"
#include "data/mesh.hh"

namespace flecsolve::vec {

template<auto V, class Topo, typename Topo::index_space Space, class Scalar>
struct mesh : base<mesh<V, Topo, Space, Scalar>> {
	using base_t = base<mesh<V, Topo, Space, Scalar>>;

	template<class Slot, class Ref>
	mesh(variable_t<V>, Slot & topo, Ref ref)
		: base_t(data::mesh<Topo, Space, Scalar>{topo, ref}) {}

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
mesh(variable_t<V>, Slot &, Ref)
	-> mesh<V, typename Ref::Topology, Ref::space, typename Ref::value_type>;

template<class Topo, typename Topo::index_space Space, class Scalar>
struct mesh<anon_var::anonymous, Topo, Space, Scalar>
	: base<mesh<anon_var::anonymous, Topo, Space, Scalar>> {
	using base_t = base<mesh<anon_var::anonymous, Topo, Space, Scalar>>;

	template<class Slot, class Ref>
	mesh(Slot & topo, Ref ref)
		: base_t(data::mesh<Topo, Space, Scalar>{std::ref(topo), ref}) {}

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
mesh(Slot &, Ref) -> mesh<anon_var::anonymous,
                          typename Ref::Topology,
                          Ref::space,
                          typename Ref::value_type>;
}

namespace flecsolve {
template<auto V, class Topo, typename Topo::index_space Space, class Scalar>
struct traits<vec::mesh<V, Topo, Space, Scalar>> {
	static constexpr auto var = variable<V>;
	using data_t = vec::data::mesh<Topo, Space, Scalar>;
	using ops_t = vec::ops::mesh<Topo, Space, Scalar>;
};
}
#endif
