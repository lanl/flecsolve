#ifndef FLECSI_LINALG_VECTORS_TOPO_VIEW_HH
#define FLECSI_LINALG_VECTORS_TOPO_VIEW_HH

#include "core.hh"
#include "variable.hh"
#include "operations/topo_view.hh"
#include "data/topo_view.hh"

namespace flecsolve::vec {

namespace detail {
template<class Topo, class Ref, typename = void>
struct is_field_reference : std::false_type {};

template<class Topo, class Ref>
struct is_field_reference<
	Topo,
	Ref,
	typename std::enable_if_t<
		std::is_same_v<typename flecsi::field<typename Ref::value_type,
                                              flecsi::data::layout::dense>::
                           template Reference<Topo, Ref::space>,
                       Ref>>> : std::true_type {};
template<class T, class R>
inline constexpr bool is_field_reference_v = is_field_reference<T, R>::value;

}

template<auto V, class Scalar, class Topo, typename Topo::index_space Space>
struct topo_view_config {
	using scalar = Scalar;
	using real = typename num_traits<scalar>::real;
	using len_t = flecsi::util::id;
	static constexpr auto var = variable<V>;
	using var_t = decltype(V);
	static constexpr std::size_t num_components = 1;
	using topo_t = Topo;
	static constexpr typename Topo::index_space space = Space;
};

template<auto V, class Scalar, class Topo, typename Topo::index_space Space>
using topo_view = core<data::topo_view,
                       ops::topo_view,
                       topo_view_config<V, Scalar, Topo, Space>>;

template<auto V,
         class Topo,
         class Ref,
         std::enable_if_t<detail::is_field_reference_v<Topo, Ref>, bool> = true>
auto make(variable_t<V>, flecsi::data::topology_slot<Topo> & topo, Ref ref) {
	using vec_t = topo_view<V, typename Ref::value_type, Topo, Ref::space>;
	using config_t = typename vec_t::config;

	return vec_t{data::topo_view<config_t>{topo, ref}};
}

template<class Topo,
         class Ref,
         std::enable_if_t<detail::is_field_reference_v<Topo, Ref>, bool> = true>
auto make(flecsi::data::topology_slot<Topo> & topo, Ref ref) {
	return make(variable<anon_var::anonymous>, topo, ref);
}

template<auto V, class Topo>
auto make(variable_t<V> var, flecsi::data::topology_slot<Topo> & topo) {
	return [&, var](auto &... fd) {
		if constexpr (sizeof...(fd) == 1) {
			return make(var, topo, fd(topo)...);
		}
		else {
			return std::tuple(make(var, topo, fd(topo))...);
		}
	};
}

template<class Topo>
auto make(flecsi::data::topology_slot<Topo> & topo) {
	return make(variable<anon_var::anonymous>, topo);
}

}
#endif
