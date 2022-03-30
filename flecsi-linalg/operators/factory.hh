#ifndef FLECSI_LINALG_OP_FACTORY_H
#define FLECSI_LINALG_OP_FACTORY_H

#include <utility>

namespace flecsi::linalg {

template <class T> struct traits {};

namespace op {

template <class T> struct factory;

template <class T>
auto create(T&& p) {
	return factory<typename traits<T>::op>::create(std::forward<T>(p));
}

}

}
#endif
