#pragma once

#include <utility>

#include "flecsi/data.hh"
#include "flecsi/util/constant.hh"

namespace flecsolve {
namespace physics {
namespace utils {

namespace mp {
// *********************
// These structures are useful for modifying type lists, e.g. list of axis
// types.
// *********************
template<auto... V>
using has = flecsi::util::constants<V...>;

template<auto A, class B>
struct cons_ {};
template<auto A0, auto... A>
struct cons_<A0, has<A...>> {
	using type = has<A0, A...>;
};
template<auto A, class B>
using cons = typename cons_<A, B>::type;

// Remove from the second list the elements of the first list. None may have
// repeated elements, but they may be unsorted.
template<class S, class T, class SS = S>
struct complement_list_;
template<class S, class T, class SS = S>
using complement_list = typename complement_list_<S, T, SS>::type;

using nil = has<>;

// end of T.
template<class S, class SS>
struct complement_list_<S, nil, SS> {
	using type = nil;
};
// end search on S, did not find.
template<auto T0, auto... T, class SS>
struct complement_list_<nil, has<T0, T...>, SS> {
	using type = cons<T0, complement_list<SS, has<T...>>>;
};
// end search on S, found.
template<auto F, auto... S, auto... T, class SS>
struct complement_list_<has<F, S...>, has<F, T...>, SS> {
	using type = complement_list<SS, has<T...>>;
};
// keep searching on S.
template<auto S0, auto... S, auto T0, auto... T, class SS>
struct complement_list_<has<S0, S...>, has<T0, T...>, SS> {
	using type = complement_list<has<S...>, has<T0, T...>, SS>;
};

template<class S>
struct rotate_;
template<class S>
using rotate = typename rotate_<S>::type;

template<auto S, auto... T>
struct rotate_<has<S, T...>> {
	using type = has<T..., S>;
};

template<auto T, class S>
struct rotate_to_;

template<auto T, class S>
using rotate_to = typename rotate_to_<T, S>::type;

template<auto S, auto... T>
struct rotate_to_<S, has<S, T...>> {
	using type = has<S, T...>;
};

template<auto S, auto P, auto... T>
struct rotate_to_<S, has<P, T...>> {
	using type = typename rotate_to_<S, has<T..., P>>::type;
};

} // namespace mp

// template<std::size_t... II>

inline static std::size_t digit(flecsi::util::id & x, std::size_t d) {
	std::size_t ret = x % d;
	x /= d;
	return ret;
}

struct srange {
	std::size_t beg, end;
	std::size_t size() const { return end - beg; }
};

inline flecsi::util::id
translate(flecsi::util::id & x, std::size_t stride, const srange & sub) {
	flecsi::util::id ret;

	ret = digit(x, sub.size()) + sub.beg;

	ret *= stride;

	return ret;
}

template<std::size_t... Index>
flecsi::util::id
translate_index(flecsi::util::id x,
                const std::array<srange, sizeof...(Index)> & subrange,
                const std::array<std::size_t, sizeof...(Index)> & strides,
                std::index_sequence<Index...>) {
	return (translate(x, strides[Index], subrange[Index]) + ...);
}

template<std::size_t... Index>
std::size_t subrange_size(std::array<srange, sizeof...(Index)> & subrange,
                          std::index_sequence<Index...>) {
	return (subrange[Index].size() * ...);
}

template<auto S, unsigned short Dim>
inline auto make_subrange_ids(std::array<srange, Dim> subrange,
                              std::array<std::size_t, Dim> strides) {
	return flecsi::util::transform_view(
		flecsi::util::iota_view<flecsi::util::id>(
			0, subrange_size(subrange, std::make_index_sequence<Dim>())),
		[=](const auto & x) {
			return flecsi::topo::id<S>(translate_index(
				x, subrange, strides, std::make_index_sequence<Dim>()));
		});
}

template<int... OFFSETS>
constexpr auto offset_seq() {
	return std::integer_sequence<int, OFFSETS...>{};
}

}
}
}
