#include "ua.hh"

namespace flecsolve::mg::ua {

po::options_description solver::options::operator()(settings_type & s) {
	po::options_description desc;
	desc.add_options()
		(label("max-levels").c_str(), po::value<int>(&s.max_levels)->required(), "Max number of levels");

	return desc;
}
solver::solver(const settings & s) : settings_(s) {}

}
