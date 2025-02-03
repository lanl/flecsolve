#include "ua.hh"

namespace flecsolve::mg::ua {

po::options_description solver::options::operator()(settings_type & s) {
	po::options_description desc;
	desc.add_options()
		(label("max-levels").c_str(), po::value<std::size_t>(&s.max_levels)->required(), "Max number of levels")
		(label("beta").c_str(), po::value<float>(&s.coarsening_settings.beta)->default_value(0.25), "Beta")
		(label("redist-coarsen-factor").c_str(), po::value<std::size_t>(&s.coarsening_settings.redist_coarsen_factor)->default_value(2), "Redistribution coarsening factor")
		(label("min-local-coarse").c_str(), po::value<std::size_t>(&s.coarsening_settings.min_local_coarse), "Minimum number of local rows before redistributing")
		(label("redistribute").c_str(), po::value<bool>(&s.coarsening_settings.redistribute)->default_value(true))
		(label("pairwise-passes").c_str(), po::value<std::size_t>(&s.coarsening_settings.pairwise_passes)->default_value(2))
		(label("min-coarse").c_str(), po::value<std::size_t>(&s.min_coarse)->default_value(5))
		(label("boomer-cg").c_str(), po::value<bool>(&s.boomer_cg)->default_value(false))
		(label("atol").c_str(), po::value<float>(&s.atol)->default_value(1e-6), "Absolute residual tolerance")
		(label("maxiter").c_str(), po::value<std::size_t>(&s.maxiter)->default_value(10))
		(label("jacobi-weight").c_str(), po::value<float>(&s.jacobi_weight)->default_value(0.6666666))
		(label("cycle").c_str(), po::value<cycle_type>(&s.cycle)->default_value(cycle_type::v), "Cycle type")
		(label("kappa").c_str(), po::value<int>(&s.kappa)->default_value(1))
		(label("ktol").c_str(), po::value<float>(&s.ktol)->default_value(0.25))
		(label("nrelax").c_str(), po::value<std::size_t>(&s.nrelax)->default_value(2));

	return desc;
}
solver::solver(const settings & s) : settings_(s) {}

}
