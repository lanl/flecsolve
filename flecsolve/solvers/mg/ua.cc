#include "ua.hh"

namespace flecsolve::mg::ua {

po::options_description solver::options::operator()(settings_type & s) {
	po::options_description desc;
	desc.add_options()
		(label("max-levels").c_str(), po::value<std::size_t>(&s.max_levels)->required(), "Max number of levels")
		(label("beta").c_str(), po::value<float>(&s.coarsening_settings.beta)->default_value(0.25), "Beta")
		(label("redist-coarsen-factor").c_str(), po::value<std::size_t>(&s.coarsening_settings.redist_coarsen_factor)->default_value(2), "Redistribution coarsening factor")
		(label("min-local-coarse").c_str(), po::value<std::size_t>(&s.coarsening_settings.min_local_coarse), "Minimum number of local rows before redistributing")
		(label("pairwise-passes").c_str(), po::value<std::size_t>(&s.coarsening_settings.pairwise_passes)->default_value(2))
		(label("min-coarse").c_str(), po::value<std::size_t>(&s.min_coarse)->default_value(5))
		(label("maxiter").c_str(), po::value<std::size_t>(&s.maxiter)->default_value(10))
		(label("jacobi-weight").c_str(), po::value<float>(&s.jacobi_weight)->default_value(0.6666666))
		(label("cycle").c_str(), po::value<cycle_type>(&s.cycle)->default_value(cycle_type::v), "Cycle type")
		(label("nrelax").c_str(), po::value<std::size_t>(&s.nrelax)->default_value(2));

	return desc;
}
solver::solver(const settings & s) : settings_(s) {}

}
