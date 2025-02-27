#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/RawCSRMatrixParameters.h"
#include "AMP/matrices/data/hypre/HypreCSRPolicy.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/testHelpers/MatrixDataTransforms.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "flecsi/flog.hh"
#include "flecsi/util/unit.hh"
#include "flecsi/util/unit/types.hh"

#include "flecsolve/matrices/parcsr.hh"
#include "flecsolve/solvers/amp.hh"

using namespace flecsolve;

using matpol = AMP::LinearAlgebra::HypreCSRPolicy;

using parcsr = mat::parcsr<matpol::scalar_t>;
using csr_topo = parcsr::topo_t;
using csr = mat::csr<matpol::scalar_t>;

csr_topo::vec_def<csr_topo::cols> ud, fd;

auto create_amp_mat(csr_topo::init & init, std::shared_ptr<AMP::Database> input_db,
                    std::shared_ptr<AMP::LinearAlgebra::Vector> & rhs,
                    std::shared_ptr<AMP::LinearAlgebra::Vector> & sol,
                    std::shared_ptr<AMP::Operator::Operator> & linearOperator) {
	AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );

    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto meshAdapter = AMP::Mesh::MeshFactory::create( mgrParams );

    int DOFsPerNode          = 1;
    int nodalGhostWidth      = 1;
    bool split               = true;
    auto nodalDofMap         = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size; MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::shared_ptr<AMP::Operator::ElementPhysicsModel> transportModel;
    linearOperator = AMP::Operator::OperatorBuilder::createOperator(
	    meshAdapter, "DiffusionBVPOperator", input_db, transportModel );
    auto diffusionOperator =
        std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>( linearOperator );

    auto boundaryOpCorrectionVec = AMP::LinearAlgebra::createVector(nodalDofMap,
                                                                    diffusionOperator->getOutputVariable());

    auto boundaryOp = diffusionOperator->getBoundaryOperator();
    boundaryOp->addRHScorrection( boundaryOpCorrectionVec );

    rhs = AMP::LinearAlgebra::createVector(nodalDofMap,
                                           diffusionOperator->getOutputVariable(),
                                           true,
                                           diffusionOperator->getMemoryLocation());
    sol = AMP::LinearAlgebra::createVector(nodalDofMap,
                                           diffusionOperator->getInputVariable(),
                                           true,
                                           diffusionOperator->getMemoryLocation());

    using Policy   = AMP::LinearAlgebra::HypreCSRPolicy;
    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    std::array<gidx_t, 2> row_rng, col_rng;
    struct split_params {
	    std::vector<lidx_t> rowptr;
	    std::vector<gidx_t> cols;
	    std::vector<scalar_t> coeffs;
    };
    struct split {
	    split_params diag, offd;
    } param_input;
    [&](split_params & diag, split_params & offd) {
	    AMP::LinearAlgebra::transformDofToCSR<Policy>(diffusionOperator->getMatrix(),
	                                                  row_rng[0], row_rng[1],
	                                                  col_rng[0], col_rng[1],
	                                                  diag.rowptr,
	                                                  diag.cols,
	                                                  diag.coeffs,
	                                                  offd.rowptr,
	                                                  offd.cols,
	                                                  offd.coeffs);
    }(param_input.diag, param_input.offd);

    auto [params_diag, params_offd] = [](auto & ... in) {
	    return std::make_pair(
		    AMP::LinearAlgebra::RawCSRMatrixParameters<Policy>::RawCSRLocalMatrixParameters{
			    in.rowptr.data(), in.cols.data(), in.coeffs.data()}...);
    }(param_input.diag, param_input.offd);

    auto csr_params = std::make_shared<AMP::LinearAlgebra::RawCSRMatrixParameters<Policy>>(
	    row_rng[0], row_rng[1], col_rng[0], col_rng[1], params_diag, params_offd, meshAdapter->getComm());
    auto csrMatrix = std::make_shared<AMP::LinearAlgebra::CSRMatrix<Policy>>(csr_params);


    using csr_data = AMP::LinearAlgebra::CSRMatrixData<Policy>;
    auto & mdata = dynamic_cast<csr_data&>(*csrMatrix->getMatrixData());

    init.comm = MPI_COMM_WORLD;
    init.ncols = init.nrows = mdata.numGlobalRows();
    auto lastrow = flecsi::util::mpi::all_gatherv(row_rng[1]);
    flecsi::util::offsets::storage store;
    std::copy(lastrow.begin(), lastrow.end(), std::back_inserter(store));
    flecsi::util::offsets rowpart(std::move(store));
    init.row_part.set_offsets(rowpart);
    init.col_part.set_offsets(rowpart);

    csr procmat(mdata.numLocalRows(), mdata.numLocalColumns());
    procmat.resize(mdata.numberOfNonZeros());

    auto [rowptr, colind, values] = procmat.rep();
    std::size_t nnz_count = 0;
    for (std::size_t i = 0; i < mdata.numLocalRows(); ++i) {
	    std::vector<double> coeffs;
	    std::vector<std::size_t> cols;
	    mdata.getRowByGlobalID(i + mdata.beginRow(), cols, coeffs);
	    std::copy(cols.begin(), cols.end(), colind.begin() + nnz_count);
	    std::copy(coeffs.begin(), coeffs.end(), values.begin() + nnz_count);
	    nnz_count += cols.size();
	    rowptr[i+1] = nnz_count;
    }

    init.proc_mats.push_back(std::move(procmat));
}


void start_amp() {
	int argc = 1;
	char *argv[] = {(char*)""};
	AMP::AMPManagerProperties props;
	props.stack_trace_type = 2;
	props.COMM_WORLD = MPI_COMM_WORLD;
	AMP::AMPManager::startup(argc, argv, props);
}


void stop_amp() { AMP::AMPManager::shutdown(); }


int finalize_amp() {
	flecsi::execute<stop_amp, flecsi::mpi>();

	return 0;
}

int amptest() {
	flecsi::execute<start_amp, flecsi::mpi>();

	std::string input_file{"amp-solver-input"};
	auto input_db = AMP::Database::parseInputFile(input_file);
	csr_topo::init init;
	std::shared_ptr<AMP::LinearAlgebra::Vector> rhs, sol;
	std::shared_ptr<AMP::Operator::Operator> linop;
	flecsi::execute<create_amp_mat,flecsi::mpi>(init, input_db, rhs, sol, linop);
	op::core<parcsr> A(std::move(init));
	auto & topo = A.data.topo();

	UNIT(){
		auto [u, f] = vec::make(topo)(ud, fd);

		auto run_directly = [&](const std::string & solver_name) {
			auto amp_solver =
				AMP::Solver::Test::buildSolver(solver_name, input_db,
				                               AMP::AMP_MPI(AMP_COMM_WORLD), nullptr, linop);
			sol->setToScalar(1.);
			linop->apply(sol, rhs);
			sol->setToScalar(0.);
			amp_solver->setZeroInitialGuess(false);
			amp_solver->apply(rhs, sol);

			return amp_solver->getIterations();
		};

		{
			u.set_scalar(1.);
			A(u, f);
			u.zero();
			auto settings = read_config("amp-solver.cfg", amp::solver::options("solver"));
			amp::solver slv{settings,
			                *input_db};
			auto info = slv(std::ref(A))(f, u);

			EXPECT_EQ(info.iters, run_directly(settings.solver_name));
			EXPECT_TRUE(info.success());
		}
		// with pcg
		{
			u.set_scalar(1.);
			A(u, f);
			u.zero();
			auto settings = read_config("amp-solver-pcg.cfg", amp::solver::options("solver"));
			amp::solver slv{settings, *input_db};
			auto info = slv(std::ref(A))(f, u);

			EXPECT_EQ(info.iters, run_directly(settings.solver_name));
			EXPECT_TRUE(info.success());
		}
		// with gmres
		{
			u.set_scalar(1.);
			A(u, f);
			u.zero();
			auto settings = read_config("amp-solver-gmres.cfg", amp::solver::options("solver"));
			amp::solver slv{settings, *input_db};
			auto info = slv(std::ref(A))(f, u);

			EXPECT_EQ(info.iters, run_directly(settings.solver_name));
			EXPECT_TRUE(info.success());
		}
	};
}

flecsi::util::unit::driver<amptest> driver;
flecsi::util::unit::finalization<finalize_amp> finalize;
