#ifndef FLECSOLVE_VECTORS_DATA_MULTI_HH
#define FLECSOLVE_VECTORS_DATA_MULTI_HH

#include <tuple>
#include <type_traits>

#include "flecsolve/util/traits.hh"
#include "flecsolve/vectors/variable.hh"

namespace flecsolve::vec {

namespace data {

template<class Config>
struct multi {
	using config = Config;
	typename Config::storage_type components;
};

}
}

#endif
