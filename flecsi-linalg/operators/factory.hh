#ifndef FLECSI_LINALG_OP_FACTORY_H
#define FLECSI_LINALG_OP_FACTORY_H

#include <utility>

#include "flecsi-linalg/util/traits.hh"

namespace flecsi::linalg::op {

template <class T> struct factory;

template <class T>
auto create(T&& p) {
	return factory<typename traits<T>::op>::create(std::forward<T>(p));
}

}
#endif
