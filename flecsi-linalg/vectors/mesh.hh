#pragma once

#include "vector.hh"
#include "variable.hh"
#include "operations/mesh.hh"
#include "data/mesh.hh"

namespace flecsi::linalg::vec {


template <auto V, class Topo, typename Topo::index_space Space, class Scalar>
struct mesh
	: vector<data::mesh<Topo, Space, Scalar>,
	         ops::mesh<Topo, Space, vector_types<Scalar>>, V> {
	using base_t = vector<data::mesh<Topo, Space, Scalar>,
	                      ops::mesh<Topo, Space, vector_types<Scalar>>, V>;

	template<class Slot, class Ref>
	mesh(variable_t<V>, Slot & topo, Ref ref) : base_t(data::mesh<Topo, Space, Scalar>{topo, ref}) {}
};

template<auto V, class Slot, class Ref>
mesh(variable_t<V>, Slot &, Ref)->mesh<V, typename Ref::Base::Topology, Ref::space, typename Ref::value_type>;


template <class Topo, typename Topo::index_space Space, class Scalar>
struct mesh<nullptr, Topo, Space, Scalar>
	: vector<data::mesh<Topo, Space, Scalar>,
	         ops::mesh<Topo, Space, vector_types<Scalar>>, nullptr> {
	using base_t = vector<data::mesh<Topo, Space, Scalar>,
	                      ops::mesh<Topo, Space, vector_types<Scalar>>, nullptr>;

	template<class Slot, class Ref>
	mesh(Slot & topo, Ref ref) : base_t(data::mesh<Topo, Space, Scalar>{topo, ref}) {}
};

template<class Slot, class Ref>
mesh(Slot &, Ref)->mesh<nullptr, typename Ref::Base::Topology, Ref::space, typename Ref::value_type>;

}
