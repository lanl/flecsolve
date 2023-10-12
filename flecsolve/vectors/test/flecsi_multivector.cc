#include <array>

#include <flecsi/flog.hh>
#include <flecsi/util/function_traits.hh>
#include <flecsi/util/unit.hh>
#include <flecsi/util/unit/types.hh>
#include <utility>

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/util/test/mesh.hh"

using namespace flecsi;

namespace flecsolve {

testmesh::slot msh;
testmesh::cslot coloring;

constexpr std::size_t nvars = 4;
using fd_array =
	std::array<field<double>::definition<testmesh, testmesh::cells>, nvars>;
const fd_array xd, yd, zd, tmpd;
using make_is = std::make_index_sequence<nvars>;

std::array<field<double>::definition<testmesh, testmesh::cells>, 3> named_defs;

template<auto V>
constexpr decltype(auto) defs() {
	return named_defs[static_cast<std::size_t>(V)];
}

enum class vars { pressure, temperature, density };

void init_mesh() {
	std::vector<flecsi::util::gid> extents{32};

	coloring.allocate(flecsi::processes(), extents);
	msh.allocate(coloring.get());
}

void init_field(testmesh::accessor<ro, ro> m,
                field<double>::accessor<wo, na> xa,
                int offset,
                std::size_t index) {
	for (auto dof : m.dofs<testmesh::cells>()) {
		xa[dof] = (offset + index + 1) * m.global_id(dof);
	}
}

template<std::size_t... Index>
void init_fields(const fd_array & arr,
                 int offset,
                 std::index_sequence<Index...>) {
	(execute<init_field>(msh, arr[Index](msh), offset, Index), ...);
}

template<std::size_t... Index>
auto create_multivector(const fd_array & arr, std::index_sequence<Index...>) {
	return vec::multi(vec::mesh(msh, arr[Index](msh))...);
}

static constexpr double ftol = 1e-8;

template<class FN>
decltype(auto) check_f(FN && fn,
                       testmesh::accessor<ro, ro> m,
                       field<double>::accessor<ro, na> x,
                       int index) {
	UNIT (fn.second) {
		for (auto dof : m.dofs<testmesh::cells>()) {
			auto gid = m.global_id(dof);
			EXPECT_LT(std::abs(fn.first(gid, index) - x[dof]), ftol);
		}
	};
}

template<class MV, class S, class FN>
bool run(MV & mv, S & msh, FN && fn) {
	int ind = 0;
	return std::apply(
		[&](auto &... v) {
			return ((test<check_f<FN>>(fn, msh, v.data.ref(), ind++) == 0) and
		            ...);
		},
		mv.data);
}

auto add = std::make_pair(
	[](std::size_t gid, int index) {
		return (index + 1) * gid + (index + 3) * gid;
	},
	"add");
auto sub = std::make_pair(
	[](double gid, double index) {
		return (index + 2) * gid - (index + 3) * gid;
	},
	"subtract");
auto mult = std::make_pair(
	[](std::size_t gid, int index) {
		return (index + 1) * gid * (index + 3) * gid;
	},
	"multiply");
auto scalar_add = std::make_pair(
	[](std::size_t gid, int index) { return (index + 1) * gid + 1; },
	"add scalar");
auto divide = std::make_pair(
	[](double gid, double index) {
		return ((index + 2) * gid) / ((index + 1) * gid + 1);
	},
	"divide");
auto scale = std::make_pair(
	[](double gid, double index) { return (index + 1) * gid * 3.5; },
	"scale");
auto recip = std::make_pair(
	[](double gid, double index) { return 1.0 / ((index + 2) * gid + 2); },
	"reciprocal");
auto linsum = std::make_pair(
	[](double gid, double index) {
		return ((index + 2) * gid) * 8 + ((index + 3) * gid) * 9;
	},
	"linear sum");

auto axpy = std::make_pair(
	[](double gid, double index) {
		return (index + 1) * gid * 7 + ((index + 2) * gid);
	},
	"axpy");
auto axpby = std::make_pair(
	[](double gid, double index) {
		return ((index + 3) * gid) * 4 + ((index + 2) * gid) * 11;
	},
	"axpby");
auto abs = std::make_pair(
	[](double gid, double index) { return std::abs((index + 2) * gid - 4.3); },
	"abs");

int vectest() {
	init_mesh();
	init_fields(xd, 0, make_is());
	init_fields(yd, 1, make_is());
	init_fields(zd, 2, make_is());
	UNIT () {
		auto x = create_multivector(xd, make_is());
		auto y = create_multivector(yd, make_is());
		auto z = create_multivector(zd, make_is());
		auto tmp = create_multivector(tmpd, make_is());

		tmp.add(x, z);
		EXPECT_TRUE(run(tmp, msh, add));

		tmp.subtract(y, z);
		EXPECT_TRUE(run(tmp, msh, sub));

		tmp.multiply(x, z);
		EXPECT_TRUE(run(tmp, msh, mult));

		x.add_scalar(x, 1);
		EXPECT_TRUE(run(x, msh, scalar_add));

		tmp.divide(y, x);
		EXPECT_TRUE(run(tmp, msh, divide));

		x.add_scalar(x, -1);

		tmp.scale(3.5, x);
		EXPECT_TRUE(run(tmp, msh, scale));

		y.add_scalar(y, 2);
		tmp.reciprocal(y);
		EXPECT_TRUE(run(tmp, msh, recip));
		y.add_scalar(y, -2);

		tmp.linear_sum(8, y, 9, z);
		EXPECT_TRUE(run(tmp, msh, linsum));

		tmp.axpy(7, x, y);
		EXPECT_TRUE(run(tmp, msh, axpy));

		tmp.copy(y);
		tmp.axpby(4, 11, z);
		EXPECT_TRUE(run(tmp, msh, axpby));

		tmp.add_scalar(y, -4.3);
		tmp.abs(tmp);
		EXPECT_TRUE(run(tmp, msh, abs));

		tmp.add_scalar(y, -7);
		EXPECT_EQ(tmp.min().get(), -7);

		{
			auto & [t0, t1, t2, t3] = tmp;
			EXPECT_EQ(tmp.max().get(),
			          std::max({t0.max().get(),
			                    t1.max().get(),
			                    t2.max().get(),
			                    t3.max().get()}));

			tmp.add_scalar(z, -43);
			EXPECT_EQ(tmp.l1norm().get(),
			          t0.l1norm().get() + t1.l1norm().get() +
			              t2.l1norm().get() + t3.l1norm().get());
			EXPECT_EQ(tmp.l2norm().get(),
			          std::sqrt(std::pow(t0.l2norm().get(), 2) +
			                    std::pow(t1.l2norm().get(), 2) +
			                    std::pow(t2.l2norm().get(), 2) +
			                    std::pow(t3.l2norm().get(), 2)));
			EXPECT_EQ(tmp.inf_norm().get(),
			          std::max({t0.inf_norm().get(),
			                    t1.inf_norm().get(),
			                    t2.inf_norm().get(),
			                    t3.inf_norm().get()}));
		}

		{ // inner product
			auto & [x0, x1, x2, x3] = x;
			auto & [y0, y1, y2, y3] = y;
			EXPECT_EQ(x.dot(y).get(),
			          (x0.dot(y0).get() + x1.dot(y1).get() + x2.dot(y2).get() +
			           x3.dot(y3).get()));
		}

		vec::mesh pvec(
			variable<vars::pressure>, msh, defs<vars::pressure>()(msh));
		vec::mesh tvec(
			variable<vars::temperature>, msh, defs<vars::temperature>()(msh));
		vec::mesh dvec(
			variable<vars::density>, msh, defs<vars::density>()(msh));

		vec::multi mv(pvec, tvec, dvec);

		auto subset = mv.subset(multivariable<vars::density, vars::pressure>);
		auto & [dvec1, pvec1] = subset;
		EXPECT_EQ(pvec.data.fid(), pvec1.data.fid());
		EXPECT_EQ(dvec.data.fid(), dvec1.data.fid());

		auto opvars = multivariable<vars::temperature, vars::density>;
		auto subset1 = mv.subset(opvars);
		auto & [tvec1, dvec2] = subset1;
		EXPECT_EQ(tvec1.data.fid(), tvec.data.fid());
		EXPECT_EQ(dvec2.data.fid(), dvec.data.fid());

		auto & tvec2 = mv.subset(variable<vars::temperature>);
		EXPECT_EQ(tvec2.data.fid(), tvec.data.fid());
	};
}

flecsi::util::unit::driver<vectest> driver;
}
