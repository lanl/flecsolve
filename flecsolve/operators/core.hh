#ifndef FLECSOLVE_OPERATORS_CORE_HH
#define FLECSOLVE_OPERATORS_CORE_HH

#include <initializer_list>
#include <memory>
#include <functional>
#include <type_traits>

#include "traits.hh"
#include "flecsolve/util/traits.hh"
#include "flecsolve/vectors/variable.hh"
#include "flecsolve/vectors/traits.hh"

namespace flecsolve::op {

template<class T>
struct shared_make {
	using store = std::shared_ptr<T>;
	template<class... Args>
	static store make(Args &&... args) {
		return std::make_shared<T>(std::forward<Args>(args)...);
	}
};

template<class T>
struct unique_make {
	using store = std::unique_ptr<T>;
	template<class... Args>
	static store make(Args &&... args) {
		return std::make_unique<T>(std::forward<Args>(args)...);
	}
};

template<class T>
struct value_make {
	using store = T;
	template<class... Args>
	static store make(Args &&... args) {
		return T(std::forward<Args>(args)...);
	}
};

template<class T, template<class> class P>
struct storage_policy {
	using store_type = typename P<T>::store;

	template<
		class Head,
		class... Tail,
		std::enable_if_t<!std::is_same_v<std::decay_t<Head>, storage_policy>,
	                     bool> = true,
		std::enable_if_t<std::is_constructible_v<T, Head, Tail...>, bool> =
			true>
	storage_policy(Head && h, Tail &&... t)
		: storage(P<T>::make(std::forward<Head>(h), std::forward<Tail>(t)...)) {
	}

	T & get() {
		if constexpr (is_smart_ptr_v<store_type>) {
			return *storage;
		}
		else
			return storage;
	}

	const T & get() const {
		if constexpr (is_smart_ptr_v<store_type>) {
			return *storage;
		}
		else
			return storage;
	}

protected:
	store_type storage;
};

template<class T>
using value_storage = storage_policy<T, value_make>;
template<class T>
using shared_storage = storage_policy<T, shared_make>;
template<class T>
using unique_storage = storage_policy<T, unique_make>;

template<class Params = std::nullptr_t,
         class ivar = variable_t<anon_var::anonymous>,
         class ovar = variable_t<anon_var::anonymous>>
struct base {
	static constexpr auto input_var = ivar{};
	static constexpr auto output_var = ovar{};
	using params_t = Params;

	template<class Head,
	         class... Tail,
	         std::enable_if_t<!std::is_same_v<std::decay_t<Head>, base>, bool> =
	             true,
	         std::enable_if_t<
				 std::is_constructible_v<params_t, Head, Tail...> ||
					 is_aggregate_initializable_v<params_t, Head, Tail...>,
				 bool> = true>
	base(Head && h, Tail &&... t)
		: params{std::forward<Head>(h), std::forward<Tail>(t)...} {}

	template<
		class Head,
		class... Tail,
		std::enable_if_t<std::is_constructible_v<params_t,
	                                             std::initializer_list<Head>,
	                                             Tail...>,
	                     bool> = true>
	base(std::initializer_list<Head> head, Tail &&... tail)
		: params{head, std::forward<Tail>(tail)...} {}

	template<class T = params_t,
	         class = std::enable_if_t<std::is_null_pointer_v<T>>>
	base() {}

	params_t & get_params() { return params; }
	const params_t & get_params() const { return params; }

	template<op::label tag, class T>
	auto get_parameters(const T &) const {
		return nullptr;
	}

	template<class T>
	void reset(const T &) const {}

	auto & get_operator() { return *this; }
	const auto & get_operator() const { return *this; }

protected:
	params_t params;
};

template<class P, template<class> class StoragePolicy = value_storage>
struct core : StoragePolicy<P> {
	static constexpr auto input_var = P::input_var;
	static constexpr auto output_var = P::output_var;
	using store = StoragePolicy<P>;
	using params_t = typename P::params_t;

	template<class Head,
	         class... Tail,
	         std::enable_if_t<
				 !std::is_same_v<std::decay_t<Head>, core<P, StoragePolicy>>,
				 bool> = true,
	         std::enable_if_t<std::is_constructible_v<store, Head, Tail...>,
	                          bool> = true>
	core(Head && h, Tail &&... t)
		: store{std::forward<Head>(h), std::forward<Tail>(t)...} {}

	template<
		class Head,
		class... Tail,
		std::enable_if_t<std::is_constructible_v<store,
	                                             std::initializer_list<Head>,
	                                             Tail...>,
	                     bool> = true>
	core(std::initializer_list<Head> h, Tail &&... t)
		: store{h, std::forward<Tail>(t)...} {}

	P & source() { return store::get(); }
	const P & source() const { return store::get(); }

	template<class D,
	         class R,
	         std::enable_if_t<is_vector_v<D>, bool> = true,
	         std::enable_if_t<is_vector_v<R>, bool> = true>
	decltype(auto) apply(const D & x, R & y) const {
		return source().apply(x, y);
	}

	template<class D,
	         class R,
	         std::enable_if_t<is_vector_v<D>, bool> = true,
	         std::enable_if_t<is_vector_v<R>, bool> = true>
	decltype(auto) operator()(const D & x, R & y) const {
		return source().apply(x, y);
	}

	template<class B,
	         class X,
	         class R,
	         std::enable_if_t<is_vector_v<B>, bool> = true,
	         std::enable_if_t<is_vector_v<X>, bool> = true,
	         std::enable_if_t<is_vector_v<R>, bool> = true>
	void residual(const B & b, const X & x, R & r) const {
		apply(x, r);

		decltype(auto) bs = b.subset(output_var);
		decltype(auto) rs = r.subset(output_var);

		rs.subtract(bs, rs);
	}

	template<auto tag, class T>
	decltype(auto) get_parameters(T && t) const {
		return source().template get_parameters<tag>(std::forward<T>(t));
	}

	template<class T>
	void reset(T && v) const {
		source().reset(std::forward<T>(v));
	}

	auto & get_operator() { return source().get_operator(); }
	const auto & get_operator() const { return source().get_operator(); }

	auto & data() { return source().data; }
	const auto & data() const { return source().data; }
};

template<class P>
auto make(P && p) {
	return core<std::decay_t<P>, value_storage>(std::forward<P>(p));
}

template<class T>
struct is_operator : std::false_type {};

template<class P, template<class> class S>
struct is_operator<core<P, S>> : std::true_type {};

template<class T>
inline constexpr bool is_operator_v = is_operator<T>::value;

}
#endif
