#include <tuple>

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/CSRMatrixParameters.h"
#include "AMP/matrices/data/hypre/HypreCSRPolicy.h"
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
#include "AMP/matrices/testHelpers/MatrixDataTransforms.h"


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

auto create_amp_mat(csr_topo::init & init) {
	std::string input_file{"amp-input"};
	auto input_db = AMP::Database::parseInputFile(input_file);

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

    std::array<gidx_t, 2> row_rng, col_rng;
    struct split_params {
	    std::vector<lidx_t> nnz;
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
	                                                  diag.nnz,
	                                                  diag.cols,
	                                                  diag.coeffs,
	                                                  offd.nnz,
	                                                  offd.cols,
	                                                  offd.coeffs);
    }(param_input.diag, param_input.offd);

    auto [params_diag, params_offd] = [](auto & ... in) {
	    return std::make_pair(
		    AMP::LinearAlgebra::CSRMatrixParameters<Policy>::CSRSerialMatrixParameters{
			    in.nnz.data(), in.cols.data(), in.coeffs.data()}...);
    }(param_input.diag, param_input.offd);

    auto csr_params = std::make_shared<AMP::LinearAlgebra::CSRMatrixParameters<Policy>>(
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


void stop_amp() {
    AMP::AMPManager::shutdown();
}

int amptest() {
	UNIT(){
		flecsi::execute<start_amp, flecsi::mpi>();
		csr_topo::init init;
		flecsi::execute<create_amp_mat,flecsi::mpi>(init);
		op::core<parcsr> A(std::move(init));
		auto & topo = A.data.topo();
		auto [u, f] = vec::make(topo)(ud, fd);

		u.set_scalar(1.);
		A(u, f);
		u.zero();
		namespace boomeramg = amp::boomeramg;
		boomeramg::solver slv{
			read_config("amp-amg.cfg",
			            boomeramg::options("solver"))};
		auto info = slv(std::ref(A))(f, u);
		EXPECT_EQ(info.iters, 14);
		EXPECT_TRUE(info.success());
	};
}

int finalize_amp() {
	flecsi::execute<stop_amp, flecsi::mpi>();

	return 0;
}

flecsi::util::unit::driver<amptest> driver;
flecsi::util::unit::finalization<finalize_amp> finalize;
