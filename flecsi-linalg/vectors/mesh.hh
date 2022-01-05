#pragma once

#include "vector.hh"
#include "operations/mesh.hh"
#include "data/mesh.hh"

namespace flecsi::linalg::vec {

template <class Topo, typename Topo::index_space Space, class Real>
struct mesh
    : vector<data::mesh<Topo, Space, Real>, ops::mesh<Topo, Space, Real>> {
	using base_t = vector<data::mesh<Topo, Space, Real>, ops::mesh<Topo, Space, Real>>;

	template<class Slot, class Ref>
	mesh(Slot & topo, Ref ref) : base_t({topo, ref}) {}
};

template<class Slot, class Ref>
mesh(Slot &, Ref)->mesh<typename Ref::Base::Topology, Ref::space, typename Ref::value_type>;

}
