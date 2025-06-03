#include <AMP/solvers/hypre/BoomerAMGSolver.h>
#include <AMP/utils/Database.h>
#include <AMP/solvers/SolverFactory.h>
#include <boost/program_options/options_description.hpp>

#include "amp.hh"

namespace flecsolve::amp {

po::options_description solver::options::operator()(settings_type & s) {
	po::options_description desc;
	//clang-format off
	desc.add_options()
		(label("amp-solver-name").c_str(), po::value<std::string>(&s.solver_name)->required(), "AMP Solver Name in Input File");
	//clang-format on

	return desc;
}

namespace boomeramg {

po::options_description options::operator()(settings_type & s) {
	po::options_description desc;
	//clang-format off
	desc.add_options()
		(label("min-iterations").c_str(), po::value<int>(&s.min_iterations)->default_value(0), "Min iterations")
		(label("max-coarse-size").c_str(), po::value<int>(&s.max_coarse_size)->default_value(32), "Max coarse size")
		(label("min-coarse-size").c_str(), po::value<int>(&s.min_coarse_size)->default_value(10), "Min coarse size")
		(label("max-levels").c_str(), po::value<int>(&s.max_levels)->default_value(10), "Max levels")
		(label("rap2").c_str(), po::value<int>(&s.rap2)->default_value(0), "rap2")
		(label("rtol").c_str(), po::value<float>(&s.rtol)->default_value(1e-9), "Relative tolerance")
		(label("maxiter").c_str(), po::value<int>(&s.maxiter)->default_value(100), "Maximum number of iterations")
		(label("relax-type").c_str(), po::value<int>(&s.relax_type)->default_value(13), "Relaxation type")
		(label("coarsen-type").c_str(), po::value<int>(&s.coarsen_type)->default_value(10), "Coarsening type")
		(label("compute-residual").c_str(), po::value<bool>(&s.compute_residual)->default_value(false), "Compute residual before/after solve")
		(label("strong-threshold").c_str(), po::value<float>(&s.strong_threshold)->default_value(0.25), "Strong threshold")
		(label("interp-type").c_str(), po::value<int>(&s.interp_type)->default_value(6), "Interpolation type")
		(label("relax-order").c_str(), po::value<int>(&s.relax_order)->default_value(0), "Relaxation order")
		(label("nrelax").c_str(), po::value<int>(&s.nrelax)->default_value(-1), "Number of relaxation sweeps")
		(label("agg-num-levels").c_str(), po::value<int>(&s.agg_num_levels)->default_value(-1), "Number of levels to aggressively coarsen.")
		(label("print-info-level").c_str(), po::value<int>(&s.print_info_level)->default_value(0), "Info level to print");
	//clang-format on

	return desc;
}

namespace {

std::shared_ptr<AMP::Database>
create_db(const boomeramg::settings & s) {
	auto ret = std::make_shared<AMP::Database>();
	auto & db = *ret;

	db.putScalar("min_iterations", s.min_iterations);
	db.putScalar("max_coarse_size", s.max_coarse_size);
	db.putScalar("min_coarse_size", s.min_coarse_size);
	db.putScalar("max_levels", s.max_levels);
	db.putScalar("rap2", s.rap2);
	db.putScalar("relative_tolerance", s.rtol);
	db.putScalar("max_iterations", s.maxiter);
	db.putScalar("print_info_level", s.print_info_level);
	db.putScalar("relax_type", s.relax_type);
	db.putScalar("coarsen_type", s.coarsen_type);
	db.putScalar("compute_residual", s.compute_residual);
	db.putScalar("strong_threshold", s.strong_threshold);
	db.putScalar("interp_type", s.interp_type);
	db.putScalar("relax_oder", s.relax_order);
	if (s.nrelax >= 0)
		db.putScalar("number_sweeps", s.nrelax);
    if (s.agg_num_levels)
      db.putScalar("agg_num_levels", s.agg_num_levels);

	return ret;
}


void init(amp_db & db,
          const settings & s) {
	db = create_db(s);
}
}

solver::solver(const settings & s) {
	flecsi::execute<init, flecsi::mpi>(db, s);
}
}
}
