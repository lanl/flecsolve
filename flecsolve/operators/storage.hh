#ifndef FLECSOLVE_OPERATORS_STORAGE_HH
#define FLECSOLVE_OPERATORS_STORAGE_HH

#include "flecsolve/util/traits.hh"
#include "flecsolve/operators/core.hh"

namespace flecsolve::op {

template<class T>
struct stored {
	using type = T;
};

template<class T, class D>
struct stored<std::unique_ptr<T, D>> {
	using type = T;
};

template<class T>
struct stored<std::shared_ptr<T>> {
	using type = T;
};
template<class T>
struct stored<std::reference_wrapper<T>> {
	using type = T;
};

template<class T>
struct storage {
	using op_type = typename stored<T>::type;
	using store_type = T;

	template<class TT,
	         std::enable_if_t<
		!std::is_base_of_v<storage, std::remove_reference_t<TT>>,bool> = true>
	storage(TT && t) : op{std::forward<TT>(t)} {}

	op_type & get() {
		if constexpr (is_reference_wrapper_v<T>) {
			return op.get();
		}
		else if constexpr (is_smart_ptr_v<T>) {
			return *op;
		}
		else
			return op;
	}

	const op_type & get() const {
		if constexpr (is_reference_wrapper_v<T>) {
			return op.get();
		}
		else if constexpr (is_smart_ptr_v<T>) {
			return *op;
		}
		else
			return op;
	}

	std::reference_wrapper<op_type> ref() const {
		return std::ref(get());
	}

protected:
	T op;
};
template<class T>
storage(T &&) -> storage<std::decay_t<T>>;

}
#endif
