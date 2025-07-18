#include <array>

#include <flecsi/exec/backend.hh>
#include <flecsi/flog.hh>
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/physics/specializations/fvm_narray.hh"

#include "test_setup.hh"

using namespace flecsi;
namespace flecsolve {
namespace physics_testing {

fld<msh::cells> xd;

template<auto Space, auto Axis>
inline void fill_box_increment(msh::accessor<ro, ro> vm,
                               field<scalar_t>::accessor<rw, na> xa,
                               scalar_t val) {
	physics::fvmtools::apply_to(
		vm.mdspan<Space>(xa),
		vm.full_range<Space, Axis>(),
		[](const auto v) { return v + 1.0; },
		val);
}

template<auto Space, auto Axis>
inline void fill_box_slopex(msh::accessor<ro, ro> vm,
                            field<scalar_t>::accessor<rw, na> xa) {
	physics::fvmtools::apply_to_with_index(
		vm.mdspan<Space>(xa),
		vm.full_range<Space, Axis>(),
		[&](const auto, const auto, const auto i) { return i; });
}

static scalar_t val_in = 3.14;
// static std::tuple fvm_apply_args{[](auto...) { return val_in + 1.0; },
//                                "fvm_apply_args"};
// static std::tuple fvm_apply_xargs{
//	[](std::size_t, std::size_t, std::size_t i) { return i; },
//	"fvm_apply_extra_args"};

auto fvm_apply_args =
	std::make_tuple([](auto...) { return val_in + 1.0; }, "fvm_apply_args");
auto fvm_apply_xargs =
	std::make_tuple([](std::size_t, std::size_t, std::size_t i) { return i; },
                    "fvm_apply_extra_args");

inline int check_apply_to(msh::topology & m) {
	UNIT ("fvm_apply") {
		// check the apply routine
		execute<fill_box_increment<msh::cells, msh::x_axis>>(m, xd(m), val_in);
		EXPECT_TRUE(fvm_run<rxcl>(fvm_apply_args, m, xd(m)));

		//		execute<fill_box_slopex<msh::cells, msh::x_axis>>(m, xd(m));
		//		EXPECT_EQ((test<fvm_check_f<fvm_apply_xargs, fvm_range_XCL>>(m,
		// xd(m))), 0);
	};
}

int fvm_mesh_test(flecsi::scheduler & s) {
	msh::ptr mptr;

	auto & m = init_mesh(s, mptr, {16, 16, 8});
	UNIT () { check_apply_to(m); };
}

util::unit::driver<fvm_mesh_test> driver;

} // namespace physics_testing
} // namespace flecsolve
