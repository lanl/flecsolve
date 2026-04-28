#include <tuple>

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/RawCSRMatrixParameters.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/operators/diffusionFD/DiffusionFD.h"
#include "AMP/operators/diffusionFD/DiffusionRotatedAnisotropicModel.h"
#include "AMP/operators/testHelpers/FDHelper.h"
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

using matpol = AMP::LinearAlgebra::DefaultHostCSRConfig;

using parcsr = mat::parcsr<matpol::scalar_t>;
using csr_topo = parcsr::topo_t;
using csr = mat::csr<matpol::scalar_t>;

csr_topo::vec_def<csr_topo::cols> ud, fd;

auto create_amp_mat(flecsi::exec::cpu s, csr_topo::init & init) {
	std::string input_file{"amp-diffusion-2d"};
	auto input_db = AMP::Database::parseInputFile(input_file);

	AMP_INSIST(input_db->keyExists("Mesh"), "Key ''Mesh'' is missing!");
	auto mesh_db = input_db->getDatabase("Mesh");
	auto racoeff_db = input_db->getDatabase("RACoefficients");

	auto mesh_params = std::make_shared<AMP::Mesh::MeshParameters>(mesh_db);
	mesh_params->setComm(AMP_COMM_WORLD);

	std::shared_ptr<AMP::Mesh::BoxMesh> mesh =
		AMP::Mesh::BoxMesh::generate(mesh_params);

	auto radiff_model = std::make_shared<
		AMP::Operator::ManufacturedRotatedAnisotropicDiffusionModel>(
		racoeff_db);

	auto PDESourceFun =
		std::bind(&AMP::Operator::RotatedAnisotropicDiffusionModel::sourceTerm,
	              &(*radiff_model),
	              std::placeholders::_1);
	auto uexactFun = std::bind(
		&AMP::Operator::RotatedAnisotropicDiffusionModel::exactSolution,
		&(*radiff_model),
		std::placeholders::_1);

	const auto opdb = std::make_shared<AMP::Database>("linearOperatorDB");
	opdb->putScalar<int>("print_info_level", 0);
	opdb->putScalar<std::string>("name", "DiffusionFDOperator");
	opdb->putDatabase("DiffusionCoefficients",
	                  radiff_model->d_c_db->cloneDatabase());

	auto op_params = std::make_shared<AMP::Operator::OperatorParameters>(opdb);
	op_params->d_name = "DiffusionFDOperator";
	op_params->d_Mesh = mesh;

	auto diffop =
		std::make_shared<AMP::Operator::DiffusionFDOperator>(op_params);

    using Policy   = AMP::LinearAlgebra::DefaultHostCSRConfig;
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
	    AMP::LinearAlgebra::transformDofToCSR<Policy>(diffop->getMatrix(),
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
	    row_rng[0], row_rng[1], col_rng[0], col_rng[1], params_diag, params_offd, AMP_COMM_WORLD);
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
    init.proc_part.set_block_map(s.launch().size, s.launch().size);

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

int amptest(flecsi::scheduler & s) {
	UNIT(){
		flecsi::execute<start_amp, flecsi::mpi>();
		csr_topo::init init;
		flecsi::execute<create_amp_mat,flecsi::mpi>(flecsi::exec::on, init);
		op::core<parcsr> A(s, std::move(init));
		auto & topo = A.data.topo();
		auto [u, f] = vec::make(topo)(ud, fd);

		u.set_scalar(1.);
		A(u, f);
		u.zero();
		namespace boomeramg = amp::boomeramg;
		boomeramg::solver slv{
			read_config("amp-amg.cfg",
			            boomeramg::options("solver"))};
		auto info = slv(op::ref(A))(f, u);
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
