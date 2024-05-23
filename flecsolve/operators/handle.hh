#ifndef FLECSOLVE_OPERATORS_HANDLE_HH
#define FLECSOLVE_OPERATORS_HANDLE_HH

#include <variant>
#include "flecsolve/util/traits.hh"
#include "flecsolve/operators/core.hh"

namespace flecsolve::op {

template<class T>
struct handle {
	using type = T;
	using var_t = std::variant<std::shared_ptr<type>,
	                           std::reference_wrapper<type>>;
	var_t store;

	constexpr type & get() const {
		return std::visit([](const auto & a) -> auto & {
			using A = std::decay_t<decltype(a)>;
			if constexpr (is_reference_wrapper_v<A>)
				return a.get();
			else return *a;
		}, store);
	}

	constexpr operator type & () const {
		return this->get();
	}
};

template<class T>
constexpr auto ref(T & o) {
	return handle<T>{std::ref(o)};
}

template<class T>
constexpr auto cref(const T & o) {
	return handle<const T>{std::ref(o)};
}

template<class T, class ... Args>
constexpr auto make_shared(Args && ... args) {
	return handle<op::core<T>>{
		std::make_shared<op::core<T>>(
			std::forward<Args>(args)...)};
}

}
#endif
