#include <algorithm>
#include <string>

#include "flecsolve/vectors/util.hh"

namespace flecsolve::vec {

std::istream & operator>>(std::istream & in, norm_type & norm) {
	std::string token;
	in >> token;

	std::transform(token.begin(),
	               token.end(),
	               token.begin(),
	               [](unsigned char c) { return std::tolower(c); });

	if (token == "inf")
		norm = norm_type::inf;
	else if (token == "l1")
		norm = norm_type::l1;
	else if (token == "l2")
		norm = norm_type::l2;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

}
