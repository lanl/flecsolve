#include <cmath>
#include <map>
#include <string>
#include <string_view>

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/solvers/krylov_interface.hh"
#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/nka.hh"

#include "flecsolve/physics/expressions/operator_expression.hh"
#include "flecsolve/physics/reaction/fkn_mechanism.hh"
#include "flecsolve/physics/reaction/arrhenius.hh"

#include "test_mesh.hh"

namespace chems {

using namespace flecsolve;

using realf = flecsi::field<double>;
testmesh::slot msh;
testmesh::cslot coloring;

const realf::definition<testmesh, testmesh::cells> xd, yd;

using flecsi::na;
using flecsi::ro;
using flecsi::wo;

std::map<std::string_view, std::size_t> sp =
	{{"X", 0}, {"Y", 1}, {"Z", 2}, {"C", 3}, {"C2", 4}, {"C3", 5}};

const double Temperature = 300.0;

inline void
init_mesh(std::size_t nelem, testmesh::slot & msh, testmesh::cslot & coloring) {
	std::vector<std::size_t> extents{nelem};
	auto colors = testmesh::distribute(flecsi::processes(), extents);
	coloring.allocate(colors, extents);
	msh.allocate(coloring.get());
}

template<class Vec>
constexpr decltype(auto) make_fkn_operator(const Vec &) {
	return physics::fkn_mechanism<Vec::var.value>::create(
		{{sp["X"], sp["Y"], sp["Z"]}});
}

template<class Vec>
decltype(auto) make_arr_operator(const Vec &) {
	std::array<std::vector<std::size_t>, 2> reactants, products;
	reactants[0] = {sp["C"], sp["C"]};
	products[0] = {sp["C2"]};

	reactants[1] = {sp["C"], sp["C2"]};
	products[1] = {sp["C3"]};

	return physics::arrhenius<Vec::var.value, 2>::create(
		{Temperature, reactants, products});
}

int driver() {
	const std::size_t NELM = sp.size();
	init_mesh(NELM, msh, coloring);

	vec::mesh x(msh, xd(msh)), y(msh, yd(msh));

	auto fkn = make_fkn_operator(x);
	auto arr = make_arr_operator(x);

	auto mech =
		op_expr(flecsolve::multivariable<decltype(x)::var.value>, fkn, arr);

	x.set_scalar(1.0);

	mech.apply(x, y);

	return 0;
}

}