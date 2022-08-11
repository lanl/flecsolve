#include <array>

#include <flecsi/exec/backend.hh>
#include <flecsi/flog.hh>
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/physics/specializations/fvm_narray.hh"

using namespace flecsi;
namespace flecsolve {
namespace physics_testing {

using scalar_t = double;

using msh = physics::fvm_narray;

const field<scalar_t>::definition<msh, msh::cells> ad;
const field<scalar_t>::definition<msh, msh::faces> bxd, byx, bzd;

msh::slot m;
msh::cslot coloring;

inline void init_mesh(const std::vector<std::size_t> & extents) {

	auto colors = msh::distribute(processes(), extents);
	coloring.allocate(colors, extents);

	msh::gbox geometry;
	geometry[msh::x_axis][0] = 0.0;
	geometry[msh::x_axis][1] = 1.0;
	geometry[msh::y_axis] = geometry[msh::x_axis];
	geometry[msh::z_axis] = geometry[msh::x_axis];

	m.allocate(coloring.get(), geometry);
}

template<auto Space, auto Axis>
inline void fill_field_lex(msh::accessor<ro, ro> vm,
                       field<scalar_t>::accessor<wo, na> xa)

{
	auto xv = vm.mdspan<Space>(xa);
	auto ranges = vm.full_range<Space, Axis>();

	for (auto k : std ::get<2>(ranges)) {
		for (auto j : std::get<1>(ranges)) {
			for (auto i : std::get<0>(ranges)) {
				scalar_t ri = i;
				scalar_t rj = j;
				scalar_t rk = k;
				xv[k][j][i] = ri + rj * vm.extents<Space, Axis>() +
				              rk * (vm.extents<Space, Axis>() *
				                   vm.extents<msh::cells, msh::y_axis>());
			}
		}
	}
}

template<auto Space, auto Axis>
inline std::size_t check_index_values(msh::accessor<ro, ro> vm,
                           field<scalar_t>::accessor<wo, na> xa)

{
	// auto xv = vm.mdspan<msh::cells>(xa);
	auto ii = vm.full_range_flat<Space, Axis>();
	int n_wrong = 0;
	for (auto i : ii) {
		if (xa[i] != i)
			n_wrong++;
	}
	return n_wrong;
}

template<auto Space, auto Axis, class F>
inline std::size_t test_index_lex(const F& f)
{
	flecsi::execute<fill_field_lex<Space, Axis>>(m, f(m));
	auto nw = flecsi::reduce<check_index_values<Space, Axis>,
							 flecsi::exec::fold::sum>(m, f(m));
	return nw.get();
}

int fvm_mesh_test() {
	UNIT () {

		init_mesh({8, 4, 4});

		std::size_t nw = 0;
		nw = test_index_lex<msh::cells, msh::x_axis>(ad);
		EXPECT_EQ(nw, 0);
		nw = test_index_lex<msh::cells, msh::y_axis>(ad);
		EXPECT_EQ(nw, 0);
		nw = test_index_lex<msh::cells, msh::z_axis>(ad);
		EXPECT_EQ(nw, 0);
		nw = test_index_lex<msh::faces, msh::x_axis>(bxd);
		EXPECT_EQ(nw, 0);
		nw = test_index_lex<msh::faces, msh::y_axis>(bxd);
		EXPECT_EQ(nw, 0);
		nw = test_index_lex<msh::faces, msh::z_axis>(bxd);
		EXPECT_EQ(nw, 0);

	};

	return 0;
}

} // namespace physics_testing
} // namespace flecsolve