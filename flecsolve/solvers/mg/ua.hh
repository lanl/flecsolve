#ifndef FLECSOLVE_SOLVERS_MG_UA_HH
#define FLECSOLVE_SOLVERS_MG_UA_HH

#include <flecsi/flog.hh>

#include "flecsolve/util/config.hh"
#include "flecsolve/solvers/mg/jacobi.hh"
#include "flecsolve/solvers/mg/level.hh"
#include "flecsolve/solvers/mg/intergrid.hh"
#include "flecsolve/solvers/mg/coarsen.hh"
#include "flecsolve/solvers/mg/cycle.hh"
#include "flecsolve/solvers/amp.hh"
#include "flecsolve/solvers/mg/cg_solve.hh"


namespace flecsolve::mg::ua {
template<class A>
struct topovec_store {
	using csr_t = typename A::policy_type;
	using topo_t = typename csr_t::topo_t;
	using field_def = typename topo_t::template vec_def<topo_t::cols>;
	static inline std::array<const field_def,
	                         static_cast<std::size_t>(veclabel::size)> vecdefs;
	topovec_store(A & a) :
		topovec_store(a, std::make_index_sequence<static_cast<std::size_t>(veclabel::size)>())
	{
		static_assert(op::is_operator_v<std::decay_t<A>>);
	}

	template<veclabel L>
	auto & get() {
		return vecs[static_cast<std::size_t>(L)];
	}

	template<veclabel L>
	const auto & get() const {
		return vecs[static_cast<std::size_t>(L)];
	}

private:
	template<std::size_t ... I>
	topovec_store(A & a, std::index_sequence<I...>) :
		vecs{vec::make(a.data.topo())(
			vecdefs[I])...}
	{
	}

	std::array<decltype(vec::make(std::declval<A>().data.topo())(vecdefs[0])),
	           static_cast<std::size_t>(veclabel::size)> vecs;
};

template<class scalar, class size>
struct hierarchy_config
{
	using scalar_type = scalar;
	using size_type = size;

	using csr_topo = topo::csr<scalar, size>;
	static inline const flecsi::field<flecsi::util::id>::definition<
		csr_topo, csr_topo::cols> aggt_def;

	using smoother = op::core<mg::bound_jacobi<scalar_type, size_type>>;
	using coarse_op = op::core<mat::parcsr<scalar, size>>;
	using coarse_smoother = op::core<mg::bound_jacobi<scalar_type, size_type>>;
	template<class ... Ops>
	using level_gen = level<tuple_opstore, topovec_store, Ops...>;
	using level_type = level_gen<op::core<mg::ua::prolong<scalar_type, size_type>>,
	                             op::core<mg::ua::restrict<scalar_type, size_type>>,
	                             coarse_op, smoother, smoother>;
};


struct solver_settings {
	int max_levels;
};


template<class scalar, class size>
struct bound_solver : op::base<> {
	using hier_type = hierarchy<hierarchy_config<scalar, size>>;

	bound_solver(op::handle<op::core<mat::parcsr<scalar, size>>> op, const solver_settings & s) :
		settings{s},
		hier{op, create_smoother(0, op), create_smoother(0, op)}
	{
		setup();
	}

	void setup() {
		for (int i = 0; i < settings.max_levels - 1; ++i) {
			auto & fine_mat = hier.get_mat(i);
			auto & fine_topo = fine_mat.data.topo();
			auto aggt_ref = hier_type::aggt_def(fine_topo);
			auto coarse_op = op::make_shared<mat::parcsr<scalar, size>>(
				mg::ua::coarsen(fine_mat, aggt_ref));
			mg::ua::intergrid_params<scalar, size> iparams{aggt_ref};
			hier.extend(
				coarse_op,
				create_smoother(i, coarse_op),
				create_smoother(i, coarse_op),
				op::make_shared<mg::ua::prolong<scalar, size>>(iparams),
				op::make_shared<mg::ua::restrict<scalar, size>>(iparams));
		}
		cg_solver.emplace(op::ref(hier.get(-1).A()));
	}

	template<class D, class R>
	void apply(const D & b, R & x) const {
		auto & ml = const_cast<hier_type&>(hier);

		float prev;
		auto & A = ml.get(0).A();
		auto & r = ml.get(0).res();
		A.residual(b, x, r);
		prev = r.l2norm().get();
		for (int i = 0; i <= 10; ++i) {
			vcycle(b, x,
			       ml, *cg_solver);

			A.residual(b, x, r);
			auto rnorm = r.l2norm().get();
			flog(info) << "[" << i << "] " << rnorm << " "
			           << rnorm / prev << std::endl;
			prev = rnorm;
		}
	}

protected:
	template<class Op>
	auto create_smoother(int, op::handle<Op> op) {
		mg::jacobi_settings s{0.6666666666, 4};
		return op::make_shared<mg::bound_jacobi<scalar, size>>(op, s);
	}
	solver_settings settings;
	hier_type hier;
	std::optional<op::core<lapack_solver<scalar, size>>> cg_solver;
};

namespace po = boost::program_options;

struct solver
{
	using settings = solver_settings;
	struct options : with_label {
		using settings_type = settings;
		explicit options(const char * pre) : with_label(pre) {}

		po::options_description operator()(settings_type & s);
	};

	solver(const settings & s);

	template<class scalar, class size>
	auto operator()(op::handle<op::core<mat::parcsr<scalar, size>>> A) {
		return op::core<bound_solver<scalar, size>>{A, settings_};
	}

private:
	settings settings_;
};

}
#endif
