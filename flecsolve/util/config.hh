#ifndef FLECSI_LINALG_UTIL_CONFIG_H
#define FLECSI_LINALG_UTIL_CONFIG_H

#include <boost/program_options.hpp>

namespace flecsolve {

struct with_label {
	with_label(const char * pre) : prefix(pre) {}
	void set_prefix(const char * pre) {
		prefix = pre;
	}
protected:
	std::string prefix;
	std::string label(const char * suf) { return {prefix + "." + suf}; }
};

template<class... Params>
void read_config(const char * fname, Params &... params) {
	namespace po = boost::program_options;

	std::vector<std::string> prev_opts{"-1"};
	int depth = 0, depth_limit = 50;
	bool done{false};
	while (!done) {
		po::options_description desc;
		([&](auto & p) { desc.add(p.options()); }(params), ...);

		po::variables_map vm;
		po::parsed_options parsed = po::parse_config_file(fname, desc, true);
		po::store(parsed, vm);
		po::notify(vm);

		std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::exclude_positional);
		if (prev_opts == opts) done = true;
		if (depth++ >= depth_limit) done = true;

		if (done) {
			po::parsed_options parsed = po::parse_config_file(fname, desc, false);
			po::store(parsed, vm);
			po::notify(vm);
		} else {
			prev_opts = opts;
		}
	}
}

}
#endif
