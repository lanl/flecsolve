#pragma once

#include "flecsolve/physics/specializations/operator_mesh.hh"
#include <flecsi/data.hh>
#include <flecsi/util/constant.hh>

#include "parameters.hh"

using namespace flecsi;

namespace heat_eqn {

using msh = flecsolve::physics::operator_mesh;

msh::slot m;
msh::cslot coloring;

field<scalar_t>::definition<msh, msh::cells> ud, und, rhsd, diffa;

util::key_array<field<scalar_t>::definition<msh, msh::faces>, msh::axes> diffb;

} // namespace heat_eqn
