#include <algorithm>
#include <istream>
#include <string>

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

std::istream & operator>>(std::istream & in, predictor & pred) {
	std::string token;
	in >> token;

	std::transform(token.begin(),
	               token.end(),
	               token.begin(),
	               [](unsigned char c) { return std::tolower(c); });

	if (token == "ab2")
		pred = predictor::ab2;
	else if (token == "leapfrog")
		pred = predictor::leapfrog;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

std::istream & operator>>(std::istream & in, strategy & strat) {
	std::string token;
	in >> token;

	std::transform(token.begin(),
	               token.end(),
	               token.begin(),
	               [](unsigned char c) { return std::tolower(c); });

	if (token == "truncation-error")
		strat = strategy::truncation_error;
	else if (token == "constant")
		strat = strategy::constant;
	else if (token == "final-constant")
		strat = strategy::final_constant;
	else if (token == "limit-relative-change")
		strat = strategy::limit_relative_change;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

std::istream & operator>>(std::istream & in, method & meth) {
	std::string token;
	in >> token;

	if (token == "BE")
		meth = method::be;
	else if (token == "CN")
		meth = method::cn;
	else if (token == "BDF2")
		meth = method::bdf2;
	else if (token == "BDF3")
		meth = method::bdf3;
	else if (token == "BDF4")
		meth = method::bdf4;
	else if (token == "BDF5")
		meth = method::bdf5;
	else if (token == "BDF6")
		meth = method::bdf6;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

std::istream & operator>>(std::istream & in, controller & cont) {
	std::string token;
	in >> token;

	if (token == "PC.4.7")
		cont = controller::pc4_7;
	else if (token == "H211b")
		cont = controller::H211b;
	else if (token == "PC11")
		cont = controller::pc11;
	else if (token == "Deadbeat")
		cont = controller::deadbeat;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

std::istream & operator>>(std::istream & in, error_scaling & s) {
	std::string tok;
	in >> tok;

	if (tok == "fixed-scaling")
		s = error_scaling::fixed_scaling;
	else if (tok == "fixed-resolution")
		s = error_scaling::fixed_resolution;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

}
