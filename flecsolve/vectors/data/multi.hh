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
