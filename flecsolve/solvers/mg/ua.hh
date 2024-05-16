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
		vecs{vec::make(a.data().topo())(
			vecdefs[I])...}
	{
	}

	std::array<decltype(vec::make(std::declval<A>().data().topo())(vecdefs[0])),
	           static_cast<std::size_t>(veclabel::size)> vecs;
};

template<class OpType>
struct hierarchy_config
{
	using scalar_type = typename OpType::policy_type::scalar;
	using size_type = typename OpType::policy_type::size;

	using csr_topo = typename OpType::policy_type::topo_t;
	static inline const flecsi::field<flecsi::util::id>::definition<
		csr_topo, csr_topo::cols> aggt_def;

	using smoother = decltype(std::declval<mg::jacobi>()(std::declval<OpType>()));
	using coarse_op = op::core<typename OpType::policy_type>;
	using coarse_smoother = decltype(std::declval<mg::jacobi>()(std::declval<coarse_op>()));
	template<class ... Ops>
	using level_gen = level<tuple_opstore, topovec_store, Ops...>;
	using level_type = level_gen<op::core<mg::ua::prolong<scalar_type, size_type>>,
	                             op::core<mg::ua::restrict<scalar_type, size_type>>,
	                             OpType, smoother, smoother>;
};


struct solver_settings {
	int max_levels;
};


template<class FineOp>
struct bound_solver {
	using hier_type = hierarchy<hierarchy_config<FineOp>>;

	bound_solver(FineOp op, const solver_settings & s) :
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
			auto coarse_op = op::make_shared(
				mg::ua::coarsen(fine_mat, aggt_ref, 0.25));
			using scalar = typename std::remove_reference_t<decltype(fine_mat)>::scalar_t;
			using size = typename std::remove_reference_t<decltype(fine_mat)>::size_t;
			mg::ua::intergrid_params<scalar, size> iparams{aggt_ref};
			hier.extend(
				coarse_op,
				create_smoother(i, coarse_op),
				create_smoother(i, coarse_op),
				op::make(mg::ua::prolong<scalar, size>{iparams}),
				op::make(mg::ua::restrict<scalar, size>{iparams}));
		}
	}

	template<class D, class R>
	void apply(const D & b, R & x) const {
		amp::boomeramg::settings cg_settings;
		cg_settings.maxiter = 25;
		cg_settings.print_info_level = 0;
		cg_settings.compute_residual = false;
		cg_settings.rtol = 1e-12;
		cg_settings.min_coarse_size = 10;
		cg_settings.strong_threshold = 0.5;
		cg_settings.relax_type = 16;
		cg_settings.coarsen_type = 10;
		cg_settings.interp_type = 17;
		cg_settings.relax_order = 0;

		auto & Ac = hier.get(-1).A();
		amp::boomeramg::solver cg_solver{cg_settings};

		auto & ml = const_cast<hier_type&>(hier);

		float prev;
		auto & A = ml.get(0).A();
		auto & r = ml.get(0).res();
		A.residual(b, x, r);
		prev = r.l2norm().get();
		for (int i = 0; i <= 10; ++i) {
			vcycle(b, x,
			       ml, cg_solver(Ac));

			A.residual(b, x, r);
			auto rnorm = r.l2norm().get();
			flog(info) << "[" << i << "] " << rnorm << " "
			           << rnorm / prev << std::endl;
			prev = rnorm;
		}
	}

protected:
	template<class Op>
	auto create_smoother(int, Op & op) {
		mg::jacobi_settings s{0.6666666666, 4};
		return mg::jacobi{s}(op);
	}
	solver_settings settings;
	hier_type hier;
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

	template<class A>
	auto operator()(A && a) {
		return bound_solver{std::forward<A>(a), settings_};
	}

private:
	settings settings_;
};

}
#endif
