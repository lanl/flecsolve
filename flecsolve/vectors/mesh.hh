#ifndef FLECSI_LINALG_VECTORS_MESH_H
#define FLECSI_LINALG_VECTORS_MESH_H

#include "vector.hh"
#include "variable.hh"
#include "operations/mesh.hh"
#include "data/mesh.hh"

namespace flecsolve::vec {

template<auto V, class Topo, typename Topo::index_space Space, class Scalar>
struct mesh : vector<data::mesh<Topo, Space, Scalar>,
                     ops::mesh<Topo, Space, Scalar>,
                     V> {
	using base_t = vector<data::mesh<Topo, Space, Scalar>,
	                      ops::mesh<Topo, Space, Scalar>,
	                      V>;

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
mesh(variable_t<V>, Slot &, Ref) -> mesh<V,
                                         typename Ref::Base::Topology,
                                         Ref::space,
                                         typename Ref::value_type>;

template<class Topo, typename Topo::index_space Space, class Scalar>
struct mesh<anon_var::anonymous, Topo, Space, Scalar>
	: vector<data::mesh<Topo, Space, Scalar>,
             ops::mesh<Topo, Space, Scalar>,
             anon_var::anonymous> {
	using base_t = vector<data::mesh<Topo, Space, Scalar>,
	                      ops::mesh<Topo, Space, Scalar>,
	                      anon_var::anonymous>;

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
                          typename Ref::Base::Topology,
                          Ref::space,
                          typename Ref::value_type>;

}

#endif
