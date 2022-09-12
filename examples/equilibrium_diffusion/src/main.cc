/*************************************
* Example: equilibrium diffusion with multivectors
*
* This solves the
*
*	-β ∇ (b ∇ u ) = 0
*
* For a 2D multivector system
*
*	MV == [v1, v2]
*
* Initially,
*
*	MV = 1.0
*
* The operator is volume diffusion, differentiated through the BCs.
* On v1, BCs are outflow; On v2, they are zero-flux.
*
*
*/

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
