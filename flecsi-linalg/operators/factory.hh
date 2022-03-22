#ifndef FLECSI_LINALG_OP_FACTORY_H
#define FLECSI_LINALG_OP_FACTORY_H

#include <utility>
#include <type_traits>

namespace flecsi::linalg::op {

template <class T> struct factory;

template <class T>
auto create(T&& p) {
	return factory<typename std::remove_reference_t<T>::op_class>::create(std::forward<T>(p));
}

}
#endif
