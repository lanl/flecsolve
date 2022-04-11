#pragma once

#include <array>
#include <flecsi/execution.hh>
#include <flecsi/util/array_ref.hh>
#include <flecsi/util/constant.hh>
#include <operators/operator_task.hh>
#include <tuple>
#include <type_traits>
#include <utility>

namespace flecsi {
namespace linalg {
namespace discrete_operators {

namespace detail {
struct final_tag;

template<class>
struct empty_mix;

template<template<class, template<class> class...> class This,
         class Final,
         template<class>
         class Base>
using find_final = Base<std::conditional_t<std::is_same_v<Final, final_tag>,
                                           This<final_tag>,
                                           Final>>;
}  // namespace detail
template<class Derived>
struct operator_traits;

template<class Derived>
struct operator_core
{
  using exact_type = Derived;
  using param_type = typename exact_type::Params;

  param_type params;

  template<class... Ps>
  constexpr static param_type getParamType(Ps&&... ps)
  {
    return param_type {ps...};
  }

  operator_core(param_type parameters) : params(std::move(parameters)) {}

  constexpr exact_type const& derived() const
  {
    return static_cast<exact_type const&>(*this);
  }

  constexpr exact_type& derived() { return static_cast<exact_type&>(*this); }

  template<class U, class V>
  constexpr decltype(auto) apply(U&& u, V&& v) const
  {
    return derived().apply(std::forward<U>(u), std::forward<V>(v));
  }

  constexpr decltype(auto) reset(param_type parameters)
  {
    params = std::move(parameters);
  }
};

template<template<class> class... Mixins>
struct make_mixins
{
  template<class Derived>
  struct templ : Mixins<Derived>...
  {
    template<class... Args>
    void exec(Args&&... args)
    {
      (Mixins<Derived>::exec(std::forward<Args>(args)...), ...);
    }
  };
};

template<class Derived = detail::final_tag,
         template<class> class MixinSet = detail::empty_mix>
struct operator_host
    : detail::find_final<operator_host, Derived, operator_core>,
      MixinSet<Derived>
{
  using base_type = detail::find_final<operator_host, Derived, operator_core>;
  using exact_type = typename base_type::exact_type;
  using param_type = typename base_type::param_type;
  using mixset_type = MixinSet<Derived>;

  friend operator_core<exact_type>;

  operator_host(param_type parameters) : base_type(parameters) {}

  template<class... Args>
  constexpr decltype(auto) exec(Args&&... args) const
  {
    mixset_type::exec(std::forward<Args>(args)...);
  }
};

template<class TopHost, class Mixes, class Tuple, std::size_t... ii>
constexpr decltype(auto) make_ophost_impl(std::index_sequence<ii...>,
                                          Tuple&& tuple)
{
  return TopHost(std::make_from_tuple<std::tuple_element_t<ii, Mixes>>(
      std::get<ii>(std::forward<Tuple>(tuple)))...);
}

template<template<template<class> class...> class HT,
         template<class>
         class... Mixes,
         class... Tuples>
constexpr decltype(auto) make_ophost(Tuples&&... tuples)
{
  static_assert(sizeof...(Mixes) == sizeof...(Tuples));
  using TopHost = HT<Mixes...>;
  return make_ophost_impl<TopHost, std::tuple<Mixes<TopHost>...>>(
      std::make_index_sequence<sizeof...(Mixes)> {},
      std::make_tuple(std::forward<Tuples>(tuples)...));
}

template<class Op, class... Ps>
inline decltype(auto) make_operator(Ps&&... ps)
{
  auto pars = Op::getParamType(std::forward<Ps>(ps)...);
  return Op(pars);
}

template<typename>
struct is_tuple : std::false_type
{};

template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type
{};

}  // namespace operators
}  // namespace linalg
}  // namespace flecsi
