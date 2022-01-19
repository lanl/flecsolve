#pragma once

#include "vector.hh"
#include "operations/mesh.hh"
#include "data/mesh.hh"

namespace flecsi::linalg::vec {

template <class Topo, typename Topo::index_space Space, class Scalar>
struct mesh
	: vector<data::mesh<Topo, Space, Scalar>,
	         ops::mesh<Topo, Space, vector_types<Scalar>>> {
	using base_t = vector<data::mesh<Topo, Space, Scalar>,
	                      ops::mesh<Topo, Space, vector_types<Scalar>>>;

	template<class Slot, class Ref>
	mesh(Slot & topo, Ref ref) : base_t({topo, ref}) {}
};

template<class Slot, class Ref>
mesh(Slot &, Ref)->mesh<typename Ref::Base::Topology, Ref::space, typename Ref::value_type>;

}
