#include <array>

#include <flecsi/flog.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>

#include "flecsi-linalg/vectors/flecsi_vector.hh"
#include "flecsi-linalg/vectors/multivector.hh"


#include "test_mesh.hh"

using namespace flecsi;

namespace flecsi::linalg {

mesh::slot msh;
mesh::cslot coloring;

constexpr std::size_t nvars = 4;
using fd_array = std::array<field<double>::definition<mesh, mesh::cells>, nvars>;
const fd_array xd, yd, zd, tmpd;

void init_mesh() {
	std::vector<std::size_t> extents{32};
	auto colors = mesh::distribute(flecsi::processes(), extents);
	coloring.allocate(colors, extents);

	msh.allocate(coloring.get());
}


void init_field(mesh::accessor<ro, ro> m,
                field<double>::accessor<wo, na> xa,
                int offset,
                std::size_t index) {
	for (auto dof : m.dofs<mesh::cells>()) {
		xa[dof] = (offset+index)*m.global_id(dof);
	}
}

template <std::size_t ... Index>
void init_fields(const fd_array & arr, int offset, std::index_sequence<Index...>) {
	(execute<init_field>(msh, arr[Index](msh), offset, Index),...);
}

void print_field(mesh::accessor<ro, ro> m,
                 field<double>::accessor<ro, na> xa) {
	for (auto dof : m.dofs<mesh::cells>()) {
		if (color() == 2)
			std::cout << xa[dof] << " vs " <<
				m.global_id(dof) << std::endl;
	}
}

template <std::size_t... Index>
auto create_multivector(const fd_array & arr, std::index_sequence<Index...>) {
	using vec = flecsi_vector<mesh, mesh::cells>;
	return multivector(vec{{arr[Index], msh}}...);
}

int vectest() {
	init_mesh();
	init_fields(xd, 0, std::make_index_sequence<nvars>());
	init_fields(yd, 1, std::make_index_sequence<nvars>());
	init_fields(zd, 2, std::make_index_sequence<nvars>());
	UNIT() {
		auto xv = create_multivector(xd, std::make_index_sequence<nvars>());
		auto yv = create_multivector(yd, std::make_index_sequence<nvars>());
		auto zv = create_multivector(zd, std::make_index_sequence<nvars>());
		auto tmpv = create_multivector(tmpd, std::make_index_sequence<nvars>());

		// auto & [a, b, c, d] = xv;
		// a.copy(b);
		// a.add(b, c);

		// tmpv.scale(3.5);
		// tmpv.zero();
		// tmpv.copy(xv);
		// tmpv.add(xv, zv);
		// tmpv.linear_sum(static_cast<double>(2.1), xv, static_cast<double>(3.5), zv);
		//tmpv.axpy(2.1, xv, zv);
		tmpv.zero();
		tmpv.add_scalar(tmpv, 2.0);
		auto ft = tmpv.inner_prod(xv);
		// auto ft = tmpv.max();
		// tmpv.min();
		auto ftval = ft.get();
		std::cout << ftval << std::endl;
		// std::cout << tmpv.min().get() << std::endl;;
		// execute<print_field>(msh, tmpd[1](msh));
	};
}

unit::driver<vectest> driver;

}
