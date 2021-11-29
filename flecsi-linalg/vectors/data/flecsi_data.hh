#pragma once

#include <flecsi/data.hh>

namespace flecsi::linalg {

template<class Topo, typename Topo::index_space Space, class Real=double>
struct flecsi_data {
	using real_t = Real;
	using topo_t = Topo;
	using topo_slot_t = flecsi::data::topology_slot<Topo>;
	using field_definition = typename flecsi::field<real_t>::template definition<Topo, Space>;
	const field_definition & def;
	topo_slot_t & topo;
	auto ref() const { return def(topo); }
	auto fid() const { return def(topo).fid(); }
};

}
