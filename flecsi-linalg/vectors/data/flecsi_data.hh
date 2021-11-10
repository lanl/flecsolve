#pragma once

#include <flecsi/data.hh>

namespace flecsi::linalg {

template<class Topo, typename Topo::index_space Space, class Real=double>
struct flecsi_data {
	using real_t = Real;
	using field_definition = typename flecsi::field<real_t>::template definition<Topo, Space>;
	const field_definition & def;
	flecsi::data::topology_slot<Topo> & topo;
	auto ref() const { return def(topo); }
	auto fid() const { return def(topo).fid(); }
};

}
