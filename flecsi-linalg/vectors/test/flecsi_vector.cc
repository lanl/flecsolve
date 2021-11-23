#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsi-linalg/vectors/flecsi_vector.hh"

#include "test_mesh.hh"

using namespace flecsi;

namespace flecsi::linalg {

mesh::slot msh;
mesh::cslot coloring;

const field<double>::definition<mesh, mesh::cells> xd, yd, zd, tmpd;

constexpr double ftol = 1e-8;

void init_mesh() {
	std::vector<std::size_t> extents{32};
	auto colors = mesh::distribute(flecsi::processes(), extents);
	coloring.allocate(colors, extents);

	msh.allocate(coloring.get());
}

void init_fields(mesh::accessor<ro, ro> m,
                 field<double>::accessor<wo, na> xa,
                 field<double>::accessor<wo, na> ya,
                 field<double>::accessor<wo, na> za) {
	for (auto dof : m.dofs<mesh::cells>()) {
		xa[dof] = m.global_id(dof);
		ya[dof] = m.global_id(dof) * 2;
		za[dof] = m.global_id(dof) * 3;
	}
}

void print_field(mesh::accessor<ro, ro> m,
                 field<double>::accessor<ro, na> xa) {
	for (auto dof : m.dofs<mesh::cells>()) {
		if (color() == 2)
			std::cout << xa[dof] << " vs " <<
				m.global_id(dof) << std::endl;
	}
}

int check_add(mesh::accessor<ro, ro> m,
              field<double>::accessor<ro, na> x) {
	UNIT () {
		for (auto dof : m.dofs<mesh::cells>()) {
			auto gid = m.global_id(dof);
			EXPECT_LT(std::abs((gid + 3*gid) - x[dof]), ftol);
		}
	};
}


int vectest() {
	using vec = flecsi_vector<mesh, mesh::cells>;

	init_mesh();
	execute<init_fields>(msh, xd(msh), yd(msh), zd(msh));

	UNIT() {
		vec x({xd, msh}), y({yd, msh}), z({zd, msh}), tmp({tmpd, msh});
		EXPECT_LT(std::abs(x.l2norm().get() - 102.05880657738459), ftol);

		tmp.add(x, z);
		EXPECT_EQ((test<check_add>(msh, tmpd(msh))), 0);
	};
}


unit::driver<vectest> driver;

}
