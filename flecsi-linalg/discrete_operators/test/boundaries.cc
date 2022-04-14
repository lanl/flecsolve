#include <array>

#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsi-linalg/vectors/mesh.hh"
#include "flecsi-linalg/vectors/multi.hh"

#include "flecsi-linalg/discrete_operators/boundary/dirichlet.hh"

#include "flecsi-linalg/discrete_operators/specializations/operator_mesh.hh"

using namespace flecsi;

namespace flecsi::linalg {
using msh = discrete_operators::operator_mesh;

constexpr std::size_t NX = 8;
constexpr std::size_t NY = 8;

msh::slot m;
msh::cslot coloring;

const field<double>::definition<msh, msh::cells> xd;
const field<double>::definition<msh, msh::cells> bd;

enum class bndvar { v1 = 1, v2 };

void init_mesh() {
  std::vector<std::size_t> extents{{NX, NY}};
  auto colors = msh::distribute(flecsi::processes(), extents);
  coloring.allocate(colors, extents);

  msh::grect geometry;
  geometry[0][0] = 0.0;
  geometry[0][1] = 1.0;
  geometry[1] = geometry[0];

  m.allocate(coloring.get(), geometry);
}

int boundary_test() {

  init_mesh();

  vec::mesh x(linalg::variable<bndvar::v1>, m, xd(m));
  vec::multi xm(x);

  auto bndry_xlo =
      discrete_operators::make_operator<discrete_operators::dirchilet<
          bndvar::v1, msh, msh::x_axis, msh::boundary_low>>(0.0);

  auto bndry_xhi =
      discrete_operators::make_operator<discrete_operators::dirchilet<
          bndvar::v1, msh, msh::x_axis, msh::boundary_high>>(0.0);

  xm.set_scalar(1.0);

  UNIT() { bndry_xlo.apply(xm, xm); };
}
} // namespace flecsi::linalg