/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
#ifndef FLECSOLVE_SOLVERS_AMP_HH
#define FLECSOLVE_SOLVERS_AMP_HH

#include <boost/program_options/options_description.hpp>
#include <cstddef>
#include <string>

#include <AMP/solvers/SolverFactory.h>
#include <AMP/solvers/hypre/BoomerAMGSolver.h>
#include <AMP/matrices/CSRConfig.h>
#include <AMP/matrices/CSRMatrix.h>
#include <AMP/matrices/RawCSRMatrixParameters.h>
#include <AMP/operators/LinearOperator.h>

#include <AMP/vectors/data/VectorData.h>
#include <AMP/vectors/data/ArrayVectorData.h>
#include "AMP/discretization/DOF_Manager.h"

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/vectors/Variable.h"

#include "AMP/vectors/VectorBuilder.h"

#include "flecsolve/matrices/parcsr.hh"
#include "flecsolve/util/config.hh"
#include "flecsolve/solvers/solver_settings.hh"
#include "flecsolve/operators/storage.hh"
#include "flecsolve/operators/handle.hh"


namespace flecsolve::amp {

using amp_mat =
	AMP::LinearAlgebra::CSRMatrix<AMP::LinearAlgebra::DefaultHostCSRConfig>;
using amp_policy = AMP::LinearAlgebra::DefaultHostCSRConfig;
struct seq_csr_storage {
	std::vector<amp_policy::lidx_t>   rowptr;
	std::vector<amp_policy::gidx_t>   colind;
	std::vector<amp_policy::scalar_t> values;
};

struct amp_storage {
	seq_csr_storage diag, offd;
};

struct csr_op_wrap : AMP::Operator::LinearOperator
{
	explicit csr_op_wrap(std::shared_ptr<AMP::Operator::OperatorParameters> p) :
		AMP::Operator::LinearOperator(p) {}

	amp_storage store;
};


template<class scalar, class size>
struct csr_task
{
	template<flecsi::privilege... PP>
	using vec_acc = typename flecsi::field<scalar>::template accessor<PP...>;
	using topo_t = typename mat::parcsr<scalar, size>::topo_t;
	using topo_acc = typename topo_t::template accessor<flecsi::ro>;

	template<class T,
		std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, scalar>, bool> = true>
	static auto adapt(flecsi::util::span<T> s,
	                  std::shared_ptr<AMP::Discretization::DOFManager> dofs) {
		auto base = const_cast<std::remove_const_t<T>*>(s.data());
		return AMP::LinearAlgebra::createVectorAdaptor("", dofs, base);
	}


	static void convert(topo_acc A,
	                    amp_storage & store,
	                    csr_op_wrap & wrap) {
		auto diag = A.diag();
		auto offd = A.offd();

		auto reserve = [](auto & src, auto & dst) {
			auto [rowptr, colind, values] = src.rep();
			dst.rowptr.reserve(rowptr.size());
			dst.colind.reserve(colind.size());
			dst.values.reserve(values.size());
			dst.rowptr.push_back(0);
		};

		reserve(diag, store.diag);
		reserve(offd, store.offd);

		// todo: avoid deep copy if types are compatible.
		for (std::size_t i = 0; i < diag.rows(); ++i) {
			auto add_row = [&](auto & src, auto & dst) {
				auto [rowptr, colind, values] = src.rep();
				for (std::size_t off = rowptr[i]; off < rowptr[i+1]; ++off) {
					auto lcol = colind[off];
					dst.colind.push_back(
						A.global_id(flecsi::topo::id<topo_t::cols>(lcol)));
					dst.values.push_back(values[off]);
				}
				dst.rowptr.push_back(rowptr[i+1]);
			};
			add_row(diag, store.diag);
			add_row(offd, store.offd);
		}

		const auto & meta = A.meta();

		auto [params_diag, params_offd] = [](auto & ... in) {
			return std::make_pair(
				AMP::LinearAlgebra::RawCSRMatrixParameters<amp_policy>::RawCSRLocalMatrixParameters{
					in.rowptr.data(), in.colind.data(), in.values.data()}...);
		}(store.diag, store.offd);

		auto csr_params = std::make_shared<AMP::LinearAlgebra::RawCSRMatrixParameters<amp_policy>>(
			meta.rows.beg, meta.rows.end + 1,
			meta.cols.beg, meta.cols.end + 1,
			params_diag, params_offd,
			AMP::AMP_MPI(meta.comm));

		auto csr_mat = std::make_shared<amp_mat>(csr_params);

		wrap.setMatrix(csr_mat);
		auto var = std::make_shared<AMP::LinearAlgebra::Variable>("");
		wrap.setVariables(var, var);
	}


	static AMP::LinearAlgebra::Matrix & get_matrix(AMP::Solver::SolverStrategy & slv) {
		auto op = slv.getOperator();
		flog_assert(op, "operator cannot be null");

		auto linop = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>(op);
		flog_assert(linop, "operator must be linear");

		auto matrix = linop->getMatrix();
		flog_assert(matrix, "matrix cannot be null");

		return *matrix;
	}

	static void apply(AMP::Solver::SolverStrategy & slv,
	                  topo_acc A,
	                  vec_acc<flecsi::rw, flecsi::ro> xa,
	                  vec_acc<flecsi::ro, flecsi::na> ba) {
		auto lsize = A.meta().rows.size();
		auto & matrix = get_matrix(slv);
		auto x = adapt(xa.span().first(lsize), matrix.getRightDOFManager());
		auto b = adapt(ba.span().first(lsize), matrix.getRightDOFManager());
		slv.apply(b, x);
	}


	static void get_solve_info(AMP::Solver::SolverStrategy & slv,
	                           solve_info & info) {
		info.iters = slv.getIterations();
		info.res_norm_final = slv.getResidualNorm().get<float>();
		if (slv.getConverged())
			info.status = solve_info::stop_reason::converged_user;
		else
			info.status = solve_info::stop_reason::diverged_breakdown;
	}
};

template<class scalar, class size>
std::shared_ptr<csr_op_wrap> make_op(mat::parcsr<scalar, size> & A)
{
	auto csr_op_db = std::make_shared<AMP::Database>();
	auto csr_op_params = std::make_shared<AMP::Operator::OperatorParameters>(csr_op_db);
	auto csr_op = std::make_shared<csr_op_wrap>(csr_op_params);
	flecsi::execute<csr_task<scalar, size>::convert, flecsi::mpi>(A.data.topo(),
	                                                              csr_op->store,
	                                                              *csr_op);

	return csr_op;
}

}


namespace flecsolve::op {

using namespace amp;

template<class Op>
struct amp_solver : op::base<>
{
	using scalar = typename Op::scalar_type;
	using size = typename Op::size_type;
	using parcsr = mat::parcsr<scalar, size>;
	using topo_t = typename parcsr::topo_t;
	using mat_ptr = std::shared_ptr<amp_mat>;
	using tasks = csr_task<scalar, size>;

	amp_solver(op::handle<Op> A,
	           std::shared_ptr<AMP::Database> input_db,
	           const std::string & solver_name) : A_(A) {
		slv_ = build_solver(input_db, solver_name, make_op(A_.get()));
	}

	amp_solver(op::handle<Op> A,
	           std::shared_ptr<AMP::Database> db) :
		A_(A) {
		auto params = std::make_shared<AMP::Solver::SolverStrategyParameters>(
			std::make_shared<AMP::Database>());
		params->d_db = db;
		params->d_comm = MPI_COMM_WORLD;
		params->d_pOperator = make_op(A_.get());
		slv_ = AMP::Solver::BoomerAMGSolver::createSolver(params);
	}

	template<class D, class R>
	solve_info apply(const D & b, R & x) const {
		solve_info info;

		flecsi::execute<tasks::apply, flecsi::mpi>(*slv_,
		                                           A_.get().data.topo(),
		                                           x.data.ref(),
		                                           b.data.ref());
		flecsi::execute<tasks::get_solve_info, flecsi::mpi>(*slv_, info);

		return info;
	}
private:
	std::shared_ptr<AMP::Solver::SolverStrategy>
	build_solver(std::shared_ptr<AMP::Database> input_db,
	             const std::string & solver_name,
	             std::shared_ptr<csr_op_wrap> op) {
		auto db = input_db->getDatabase(solver_name);
		auto uses_precond = db->getWithDefault<bool>("uses_preconditioner", false);
		std::shared_ptr<AMP::Solver::SolverStrategy> pc_solver;
		if (uses_precond) {
			auto pc_name = db->getWithDefault<std::string>("pc_name", "Preconditioner");
			pc_solver = build_solver(input_db, pc_name, op);
		}
		auto params = std::make_shared<AMP::Solver::SolverStrategyParameters>(db);
		params->d_comm = MPI_COMM_WORLD;
		params->d_pNestedSolver = pc_solver;
		params->d_pOperator = op;
		return AMP::Solver::SolverFactory::create(params);
	}

	op::handle<Op> A_;
	std::shared_ptr<AMP::Solver::SolverStrategy> slv_;
};
template<class O>
amp_solver(op::handle<O>,
           std::shared_ptr<AMP::Database>) -> amp_solver<O>;
template<class O>
amp_solver(op::handle<O>,
           std::shared_ptr<AMP::Database>,
           const std::string &) -> amp_solver<O>;
}

namespace flecsolve::amp {

namespace po = boost::program_options;

struct solver {
	struct settings {
		std::string solver_name;
	};

	struct options : with_label {
		using settings_type = settings;
		explicit options(const char * pre) : with_label(pre) {}

		po::options_description operator()(settings_type & s);
	};

	solver(const settings & s, std::shared_ptr<AMP::Database> db) :
		input_db{db}, solver_name{s.solver_name} {}

	template<class A>
	auto operator()(op::handle<A> a) {
		return op::make(
			op::amp_solver(a, input_db, solver_name));
	}

protected:
	std::shared_ptr<AMP::Database> input_db;
	std::string solver_name;
};


namespace boomeramg {

using amp_db = std::shared_ptr<AMP::Database>;
struct settings {
	int min_iterations;
	int max_coarse_size;
	int min_coarse_size;
	int max_levels;
	int rap2;
	float rtol;
	int maxiter;
	int print_info_level;
	int relax_type;
	int coarsen_type;
	bool compute_residual;
	float strong_threshold;
	int interp_type;
	int relax_order;
	int nrelax;
	int agg_num_levels;
};


struct options : with_label {
	using settings_type = settings;
	explicit options(const char * pre) : with_label(pre) {}

	po::options_description operator()(settings_type & s);
};


struct solver {
	solver(const settings &);

	template<class A>
	auto operator()(op::handle<A> a) {
		return op::make(
			op::amp_solver{a, db});
	}

private:
	amp_db db;
};
}
}

#endif
