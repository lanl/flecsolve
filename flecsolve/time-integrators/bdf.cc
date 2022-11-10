#include "flecsolve/time-integrators/bdf.hh"

namespace flecsolve::time_integrator::bdf {

int memory_size(method meth) {
	auto v = static_cast<std::size_t>(meth);
	return (v == 0) ? 1 : v;
}

short order(method meth) {
	switch (meth) {
		case method::cn:
			return 2;
			break;
		case method::be:
			return 1;
			break;
		default:
			return static_cast<short>(meth);
	}
}

}
