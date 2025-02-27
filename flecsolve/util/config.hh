/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
#ifndef FLECSI_LINALG_UTIL_CONFIG_H
#define FLECSI_LINALG_UTIL_CONFIG_H

#include <boost/program_options.hpp>
#include <flecsi/util/constant.hh>

namespace flecsolve {

struct with_label {
	with_label(const char * pre) : prefix(pre) {}
	void set_prefix(const char * pre) { prefix = pre; }

protected:
	std::string prefix;
	std::string label(const char * suf) { return {prefix + "." + suf}; }
};

template<auto V>
struct null_settings {};

template<auto V>
struct null_options : with_label {
	explicit null_options(const char * pre) : with_label(pre) {}
	auto operator()(null_settings<V>) {
		return boost::program_options::options_description{};
	}
};

template<auto... V>
using includes = flecsi::util::constants<V...>;

template<class... Options>
auto read_config(const char * fname, Options &&... ops) {
	namespace po = boost::program_options;
	std::tuple<typename Options::settings_type...> settings;

	std::vector<std::string> prev_opts{"-1"};
	int depth = 0, depth_limit = 50;
	bool done = false;
	while (!done) {
		po::options_description desc;

		std::apply(
			[&](auto &... spack) {
				([&](auto & o, auto & s) { desc.add(o(s)); }(ops, spack), ...);
			},
			settings);

		po::variables_map vm;
		po::parsed_options parsed = po::parse_config_file(fname, desc, true);
		po::store(parsed, vm);
		po::notify(vm);

		std::vector<std::string> opts =
			po::collect_unrecognized(parsed.options, po::exclude_positional);
		if (prev_opts == opts)
			done = true;
		if (depth++ >= depth_limit)
			done = true;

		if (done) {
			po::parsed_options parsed =
				po::parse_config_file(fname, desc, false);
			po::store(parsed, vm);
			po::notify(vm);
		}
		else {
			prev_opts = opts;
		}
	}

	if constexpr (sizeof...(ops) == 1)
		return std::get<0>(settings);
	else
		return settings;
}

}
#endif
