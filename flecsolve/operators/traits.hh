#ifndef FLECSOLVE_OP_TRAITS_H
#define FLECSOLVE_OP_TRAITS_H

#include <cstddef>

#include "flecsolve/vectors/variable.hh"

namespace flecsolve::op {

enum class label { jacobian };

template<class T>
struct traits {
	static constexpr auto input_var = variable<anon_var::anonymous>;
	static constexpr auto output_var = variable<anon_var::anonymous>;
	using parameters = std::nullptr_t;
};

}

#endif
