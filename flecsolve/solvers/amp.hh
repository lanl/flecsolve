#ifndef FLECSOLVE_SOLVERS_AMP_HH
#define FLECSOLVE_SOLVERS_AMP_HH

#include <boost/program_options/options_description.hpp>
#include <cstddef>
#include <string>

#include <AMP/solvers/hypre/BoomerAMGSolver.h>
#include <AMP/matrices/data/hypre/HypreCSRPolicy.h>
#include <AMP/matrices/CSRMatrix.h>
#include <AMP/matrices/CSRMatrixParameters.h>
#include <AMP/operators/LinearOperator.h>

#include <AMP/vectors/data/VectorData.h>
#include <AMP/vectors/data/ArrayVectorData.h>
#include <AMP/vectors/operations/VectorOperationsDefault.h>
#include "AMP/discretization/DOF_Manager.h"

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/utils/typeid.h"
#include "AMP/vectors/CommunicationList.h"
#include "AMP/vectors/Variable.h"

#include "flecsolve/matrices/parcsr.hh"
#include "flecsolve/util/config.hh"
#include "flecsolve/solvers/solver_settings.hh"

namespace flecsolve::amp {

template<class T>
struct span_vector_data : AMP::LinearAlgebra::VectorData {
	span_vector_data(flecsi::util::span<T> s, MPI_Comm comm) : vals{s} {
		setCommunicationList(std::make_shared<AMP::LinearAlgebra::CommunicationList>(s.size(), AMP::AMP_MPI(comm)));
		d_localStart = d_CommList->getStartGID();
		d_globalSize = d_CommList->getTotalSize();
		d_localSize = s.size();
	}

	std::string VectorDataName() const override { return "span_vector_data"; }

	std::size_t numberOfDataBlocks() const override { return 1; }

	std::size_t sizeOfDataBlock(std::size_t) const override { return vals.size(); }

	void addValuesByLocalID(size_t N, const size_t *, const void *, const AMP::typeID & ) override
    {
        AMP_INSIST( N == 0, "Not yet implemented" );
    }

	void getValuesByLocalID(size_t num, const size_t * indices, void * v, const AMP::typeID & id) const override
    {
	    auto get = [=](auto data) {
		    for (size_t i = 0; i != num; ++i) {
			    data[i] = vals[indices[i]];
		    }
	    };
	    if (id == AMP::getTypeID<T>()) {
		    auto data = reinterpret_cast<T*>(v);
		    get(data);
	    } else if (id == AMP::getTypeID<double>()) {
		    auto data = reinterpret_cast<double *>(v);
		    get(data);
	    } else {
		    AMP_ERROR("Conversion not supported");
	    }
    }

	void setValuesByLocalID( size_t num, const size_t * indices, const void * v, const AMP::typeID & id) override
    {
	    auto set = [=](auto data) {
		    for (size_t i = 0; i != num; ++i) {
			    vals[indices[i]] = data[i];
		    }
	    };
	    if (id == AMP::getTypeID<T>()) {
		    auto data = reinterpret_cast<const T *>(v);
		    set(data);
	    } else if (id == AMP::getTypeID<double>()) {
		    auto data = reinterpret_cast<const double *>(v);
		    set(data);
	    } else {
		    AMP_ERROR("Conversion not supported");
	    }
	    if (*d_UpdateState == AMP::LinearAlgebra::UpdateState::UNCHANGED)
		    *d_UpdateState = AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED;
    }

    template<class TYPE>
    void setLocalValuesByGlobalID( size_t num, const size_t *indices, const TYPE *vals );

	inline void putRawData( const void *, const AMP::typeID & ) override {}
	inline void getRawData( void *, const AMP::typeID & ) const override {}

    inline uint64_t getDataID() const override { return 0; }

    /** \brief Return a pointer to a particular block of memory in the vector
     */
    inline void *getRawDataBlockAsVoid( size_t ) override {
	    return vals.data();
    }

    /** \brief Return a pointer to a particular block of memory in the
     * vector
     */
	inline const void *getRawDataBlockAsVoid( size_t ) const override { return vals.data(); }

    /** \brief Return the result of sizeof(TYPE) for the given data block
     */
    inline size_t sizeofDataBlockType( size_t ) const override { return sizeof(T); }

	AMP::typeID getType( size_t ) const override
    {
	    constexpr auto type = AMP::getTypeID<T>();
        return type;
    }

    inline void swapData( VectorData & ) override { AMP_ERROR( "Not finished" ); }

    inline std::shared_ptr<VectorData> cloneData() const override
    {
	    return AMP::LinearAlgebra::ArrayVectorData<T>::create(vals.size());
    }
protected:
	flecsi::util::span<T> vals;
};

using amp_mat =
	AMP::LinearAlgebra::CSRMatrix<AMP::LinearAlgebra::HypreCSRPolicy>;
using amp_policy = AMP::LinearAlgebra::HypreCSRPolicy;
struct amp_storage {
	std::vector<amp_policy::lidx_t>   rownnz;
	std::vector<amp_policy::gidx_t>   colind;
	std::vector<amp_policy::scalar_t> values;
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
	template<flecsi::partition_privilege_t... PP>
	using vec_acc = typename flecsi::field<scalar>::template accessor<PP...>;
	using topo_t = typename mat::parcsr<scalar, size>::topo_t;
	using topo_acc = typename topo_t::template accessor<flecsi::ro>;

	template<class T,
		std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, scalar>, bool> = true>
	static auto adapt(flecsi::util::span<T> s, MPI_Comm comm) {
		auto var = std::make_shared<AMP::LinearAlgebra::Variable>("");
		auto data = [&]() {
			using vdt = span_vector_data<scalar>;
			if constexpr (std::is_const_v<T>)
				return std::make_shared<vdt>(
					flecsi::util::span<scalar>(const_cast<scalar*>(s.data()), s.size()), comm);
			else
				return std::make_shared<vdt>(s, comm);
		}();
		auto ops = std::make_shared<AMP::LinearAlgebra::VectorOperationsDefault<scalar>>();
		auto dofs = std::make_shared<AMP::Discretization::DOFManager>(s.size(), AMP::AMP_MPI(comm));
		auto vec = std::make_shared<AMP::LinearAlgebra::Vector>(data, ops, var, dofs);

		return vec;
	}


	static void convert(topo_acc A,
	                    amp_storage & store,
	                    csr_op_wrap & wrap) {
		auto diag = A.diag();
		auto offd = A.offd();

		auto [drowptr, dcolind, dvalues] = diag.rep();
		auto [orowptr, ocolind, ovalues] = offd.rep();

		store.rownnz.reserve(drowptr.size());
		store.colind.reserve(dcolind.size()); // est.
		store.values.reserve(dvalues.size() + ovalues.size());

		for (std::size_t i = 0; i < diag.rows(); ++i) {
			std::size_t rnnz = 0;
			auto add_row = [&](auto & rowptr, auto & colind, auto & values) {
				for (std::size_t off = rowptr[i]; off < rowptr[i+1]; ++off) {
					store.colind.push_back(
						A.global_id(flecsi::topo::id<topo_t::cols>(colind[off])));
					store.values.push_back(values[off]);
					++rnnz;
				}
			};
			add_row(drowptr, dcolind, dvalues);
			if (offd.nnz()) {
				add_row(orowptr, ocolind, ovalues);
			}
			store.rownnz.push_back(rnnz);
		}

		const auto & meta = A.meta();
		auto csr_params = std::make_shared<AMP::LinearAlgebra::CSRMatrixParameters<amp_policy>>(
			meta.rows.beg, meta.rows.end + 1,
			store.rownnz.data(), store.colind.data(), store.values.data(),
			AMP::AMP_MPI(meta.comm));

		auto csr_mat = std::make_shared<amp_mat>(csr_params);

		wrap.setMatrix(csr_mat);
		auto var = std::make_shared<AMP::LinearAlgebra::Variable>("");
		wrap.setVariables(var, var);
	}


	static void apply(AMP::Solver::SolverStrategy & slv,
	                  topo_acc A,
	                  vec_acc<flecsi::rw, flecsi::ro> xa,
	                  vec_acc<flecsi::ro, flecsi::na> ba) {
		auto lsize = A.meta().rows.size();
		auto comm = A.meta().comm;
		auto x = adapt(xa.span().first(lsize), comm);
		auto b = adapt(ba.span().first(lsize), comm);
		slv.apply(b, x);
	}


	static void get_solve_info(AMP::Solver::SolverStrategy & slv,
	                           solve_info & info) {
		info.iters = slv.getIterations();
		info.res_norm_final = slv.getResidualNorm().get<float>();
		if (slv.getConvergenceStatus())
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


namespace po = boost::program_options;

template<class Scalar, class Size, template<class> class storage>
struct bound_solver : op::base<>
{
	using scalar = Scalar;
	using size = Size;
	using parcsr = mat::parcsr<scalar, size>;
	using topo_t = typename parcsr::topo_t;
	using mat_ptr = std::shared_ptr<amp_mat>;
	using tasks = csr_task<scalar, size>;
	using op_t = op::core<parcsr, storage>;

	bound_solver(op_t A,
	             std::shared_ptr<AMP::Solver::SolverStrategy> slv) :
		A_(A), slv_(slv) {
		register_operator(*slv_, make_op(A_.source()));
	}

	template<class D, class R>
	solve_info apply(const D & b, R & x) const {
		solve_info info;

		flecsi::execute<tasks::apply, flecsi::mpi>(*slv_,
		                                           A_.data().topo(),
		                                           x.data.ref(),
		                                           b.data.ref());
		flecsi::execute<tasks::get_solve_info, flecsi::mpi>(*slv_, info);

		return info;
	}
private:
	void register_operator(AMP::Solver::SolverStrategy & slv,
	                       std::shared_ptr<csr_op_wrap> op) {
		slv.registerOperator(op);
		auto nest = slv.getNestedSolver();
		if (nest) register_operator(*nest, op);
	}

	op_t A_;
	std::shared_ptr<AMP::Solver::SolverStrategy> slv_;
};

struct solver {
	struct settings {
		std::string solver_name;
	};

	struct options : with_label {
		using settings_type = settings;
		explicit options(const char * pre) : with_label(pre) {}

		po::options_description operator()(settings_type & s);
	};

	solver(const settings &, AMP::Database &);

	template<class A>
	auto operator()(A && a) {
		return op::make(bound_solver{std::forward<A>(a), slv_});
	}

protected:
	std::shared_ptr<AMP::Solver::SolverStrategy> slv_;
};


namespace boomeramg {

using amp_slv = std::shared_ptr<AMP::Solver::BoomerAMGSolver>;

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
};


struct options : with_label {
	using settings_type = settings;
	explicit options(const char * pre) : with_label(pre) {}

	po::options_description operator()(settings_type & s);
};


struct solver {
	solver(const settings &);

	template<class A>
	auto operator()(A && a) {
		return op::make(bound_solver{std::forward<A>(a), slv_});
	}

private:
	amp_slv slv_;
};
}
}

#endif
