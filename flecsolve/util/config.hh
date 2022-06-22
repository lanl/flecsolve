#ifndef FLECSI_LINALG_UTIL_CONFIG_H
#define FLECSI_LINALG_UTIL_CONFIG_H

#include <boost/program_options.hpp>

namespace flecsolve {

template<class... Params>
void read_config(const char * fname, Params &... params) {
	namespace po = boost::program_options;

	po::options_description desc;
	([&](auto & p) { desc.add(p.options()); }(params), ...);

	po::variables_map vm;
	po::store(po::parse_config_file(fname, desc), vm);
	po::notify(vm);
}

}
#endif
