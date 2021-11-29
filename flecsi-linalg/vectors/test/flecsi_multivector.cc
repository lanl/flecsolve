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
using make_is = std::make_index_sequence<nvars>;

constexpr double ftol = 1e-8;

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
		xa[dof] = (offset+index+1)*m.global_id(dof);
	}
}

template <std::size_t ... Index>
void init_fields(const fd_array & arr, int offset, std::index_sequence<Index...>) {
	(execute<init_field>(msh, arr[Index](msh), offset, Index),...);
}

template <std::size_t... Index>
auto create_multivector(const fd_array & arr, std::index_sequence<Index...>) {
	using vec = flecsi_vector<mesh, mesh::cells>;
	return multivector(vec{{arr[Index], msh}}...);
}

int check_add(mesh::accessor<ro, ro> m,
              field<double>::accessor<ro, na> x,
              int index) {
	UNIT() {
		for (auto dof : m.dofs<mesh::cells>()) {
			auto gid = m.global_id(dof);
			EXPECT_LT(std::abs(((index+1)*gid + (index + 2 + 1)*gid) - x[dof]), ftol);
		}
	};
}

int vectest() {
	init_mesh();
	init_fields(xd, 0, make_is());
	init_fields(yd, 1, make_is());
	init_fields(zd, 2, make_is());
	UNIT() {
		auto x = create_multivector(xd, make_is());
		auto y = create_multivector(yd, make_is());
		auto z = create_multivector(zd, make_is());
		auto tmp = create_multivector(tmpd, make_is());

		tmp.add(x, z);
		{
			int ind = 0;
			EXPECT_TRUE(std::apply([&ind](auto & ... v) {
				return ((test<check_add>(msh, v.data.ref(),
				                         ind++) == 0) and ...);
			}, tmp.data));
		}

		{ // inner product
			auto & [x0, x1, x2, x3] = x;
			auto & [y0, y1, y2, y3] = y;
			EXPECT_EQ(x.inner_prod(y).get(),
			          (x0.inner_prod(y0).get() +
			           x1.inner_prod(y1).get() +
			           x2.inner_prod(y2).get() +
			           x3.inner_prod(y3).get()));
		}
	};
}

unit::driver<vectest> driver;

}
