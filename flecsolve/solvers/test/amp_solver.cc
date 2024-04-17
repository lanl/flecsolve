#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRMatrixParameters.h"
#include "AMP/matrices/data/hypre/HypreCSRPolicy.h"
#include "AMP/matrices/testHelpers/MatrixDataTransforms.h"
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
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/discretization/DOF_Manager.h"
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

auto create_amp_mat(csr_topo::init & init, std::shared_ptr<AMP::Database> input_db) {
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
    auto linearOperator = AMP::Operator::OperatorBuilder::createOperator(
        meshAdapter, "DiffusionBVPOperator", input_db, transportModel );
    auto diffusionOperator =
        std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>( linearOperator );

    auto boundaryOpCorrectionVec = AMP::LinearAlgebra::createVector(nodalDofMap,
                                                                    diffusionOperator->getOutputVariable());

    auto boundaryOp = diffusionOperator->getBoundaryOperator();
    boundaryOp->addRHScorrection( boundaryOpCorrectionVec );

    using Policy   = AMP::LinearAlgebra::HypreCSRPolicy;
    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    gidx_t firstRow, endRow;
    std::vector<lidx_t> nnz;
    std::vector<gidx_t> cols;
    std::vector<scalar_t> coeffs;

    AMP::LinearAlgebra::transformDofToCSR<AMP::LinearAlgebra::HypreCSRPolicy>(
        diffusionOperator->getMatrix(), firstRow, endRow, nnz, cols, coeffs );

    auto & mdata = *diffusionOperator->getMatrix()->getMatrixData();

    init.comm = MPI_COMM_WORLD;
    init.ncols = init.nrows = mdata.numGlobalRows();
    auto lastrow = flecsi::util::mpi::all_gatherv(endRow);
    flecsi::util::offsets::storage store;
    std::copy(lastrow.begin(), lastrow.end(), std::back_inserter(store));
    flecsi::util::offsets rowpart(std::move(store));
    init.row_part.set_offsets(rowpart);
    init.col_part.set_offsets(rowpart);

    csr procmat(nnz.size(), nnz.size());
    procmat.resize(coeffs.size());

    auto [rowptr, colind, values] = procmat.rep();
    std::copy(cols.begin(), cols.end(), colind.begin());
    std::copy(coeffs.begin(), coeffs.end(), values.begin());

    for (std::size_t i = 0; i < nnz.size(); ++i) {
	    rowptr[i+1] = rowptr[i] + nnz[i];
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

    AMP::Solver::registerSolverFactories();
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
	flecsi::execute<create_amp_mat,flecsi::mpi>(init, input_db);
	op::core<parcsr, op::shared_storage> A(std::move(init));
	auto & topo = A.data().topo();

	UNIT(){
		auto [u, f] = vec::make(topo)(ud, fd);
		{
			u.set_scalar(1.);
			A(u, f);
			u.zero();
			amp::solver slv{read_config("amp-solver.cfg", amp::solver::options("solver")),
			                *input_db};
			auto info = slv(A)(f, u);
			ASSERT_EQ(info.iters, 14);
			ASSERT_TRUE(info.success());
		}
		// with pcg
		{
			u.set_scalar(1.);
			A(u, f);
			u.zero();
			amp::solver slv{read_config("amp-solver-pcg.cfg", amp::solver::options("solver")),
			                *input_db};
			auto info = slv(A)(f, u);
			EXPECT_EQ(info.iters, 8);
			EXPECT_TRUE(info.success());
		}
		// with gmres
		{
			u.set_scalar(1.);
			A(u, f);
			u.zero();
			amp::solver slv{read_config("amp-solver-gmres.cfg", amp::solver::options("solver")),
			                *input_db};
			auto info = slv(A)(f, u);
			EXPECT_EQ(info.iters, 8);
		}
	};
}

flecsi::util::unit::driver<amptest> driver;
flecsi::util::unit::finalization<finalize_amp> finalize;
