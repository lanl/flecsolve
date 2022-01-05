#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsi-linalg/vectors/mesh.hh"

#include "test_mesh.hh"

using namespace flecsi;

namespace flecsi::linalg {

testmesh::slot msh;
testmesh::cslot coloring;

const field<double>::definition<testmesh, testmesh::cells> xd, yd, zd, tmpd;

constexpr double ftol = 1e-8;

void init_mesh() {
	std::vector<std::size_t> extents{32};
	auto colors = testmesh::distribute(flecsi::processes(), extents);
	coloring.allocate(colors, extents);

	msh.allocate(coloring.get());
}

void init_fields(testmesh::accessor<ro, ro> m,
                 field<double>::accessor<wo, na> xa,
                 field<double>::accessor<wo, na> ya,
                 field<double>::accessor<wo, na> za) {
	for (auto dof : m.dofs<testmesh::cells>()) {
		xa[dof] = m.global_id(dof);
		ya[dof] = m.global_id(dof) * 2;
		za[dof] = m.global_id(dof) * 3;
	}
}

int check_add(testmesh::accessor<ro, ro> m,
              field<double>::accessor<ro, na> x) {
	UNIT () {
		for (auto dof : m.dofs<testmesh::cells>()) {
			auto gid = m.global_id(dof);
			EXPECT_LT(std::abs((gid + 3*gid) - x[dof]), ftol);
		}
	};
}


int vectest() {
	init_mesh();
	execute<init_fields>(msh, xd(msh), yd(msh), zd(msh));

	UNIT() {
		vec::mesh x(msh, xd(msh)), y(msh, yd(msh)), z(msh, zd(msh)), tmp(msh, tmpd(msh));
		EXPECT_LT(std::abs(x.l2norm().get() - 102.05880657738459), ftol);

		tmp.add(x, z);
		EXPECT_EQ((test<check_add>(msh, tmpd(msh))), 0);
	};
}


unit::driver<vectest> driver;

}
