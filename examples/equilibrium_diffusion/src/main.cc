#include "equilibrium_diffusion.hh"

#include <flecsi/execution.hh>
#include <flecsi/flog.hh>
#include <flecsi/util/annotation.hh>

int main(int argc, char ** argv) {
	auto status = flecsi::initialize(argc, argv);

	if (status != flecsi::run::status::success) {
		return status < flecsi::run::status::help ? 0 : status;
	}

	flecsi::flog::add_output_stream("clog", std::clog, true);

	status = flecsi::start(eqdiff::driver);

	flecsi::finalize();

	return status;
} // main
