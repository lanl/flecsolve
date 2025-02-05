#ifndef FLECSOLVE_OPERATOR_FACTORY_H
#define FLECSOLVE_OPERATOR_FACTORY_H

#include <variant>
#include <optional>
#include <boost/program_options/options_description.hpp>

#include "flecsolve/util/config.hh"
#include "core.hh"

namespace flecsolve::op {

template<class Variant>
struct factory_prod {
	static constexpr auto input_var =
		std::variant_alternative_t<0, Variant>::input_var;
	static constexpr auto output_var =
		std::variant_alternative_t<0, Variant>::output_var;
	using params_t = std::nullptr_t;

	template<class D, class R>
	decltype(auto) apply(const D & x, R & y) const {
		return std::visit([&](auto && p) { return p.apply(x, y); }, var);
	}

	auto & get_operator() {
		return std::visit(
			[&](auto && p) -> auto & { return p.get_operator(); }, var);
	}

	const auto & get_operator() const {
		return std::visit(
			[&](auto && p) -> const auto & { return p.get_operator(); }, var);
	}

	Variant var;
};

namespace detail {
template<class P, class T>
struct var {};
template<class P, auto... V>
struct var<P, includes<V...>> {
	using settings =
		std::variant<typename P::template registry<V>::settings...>;
};

template<class... Vs>
struct var_cat;

template<class... A>
struct var_cat<std::variant<A...>> {
	using type = std::variant<A...>;
};

template<class... A, class... B, class... Rest>
struct var_cat<std::variant<A...>, std::variant<B...>, Rest...> {
	using type = typename var_cat<std::variant<A..., B...>, Rest...>::type;
};

template<class... Vs>
using var_cat_t = typename var_cat<Vs...>::type;
}

template<class Target, class Targets, class SettingsTypes>
struct factory_config {
	using target = Target;
	using targets = Targets;
	using settings_types = SettingsTypes;

	using optdesc = boost::program_options::options_description;
	struct settings {
		std::optional<target> target_id;
		settings_types target_settings;
	};
	template<class P>
	struct options : with_label {
		using settings_type = settings;
		explicit options(const char * pre) : with_label(pre) {}

		auto operator()(settings_type & s) {
			optdesc desc;

			desc.add_options()(
				label("type").c_str(),
				boost::program_options::value<target>()->required()->notifier(
					[&](const target & reg) { s.target_id.emplace(reg); }),
				"operator type");

			if (s.target_id.has_value()) {
				s.target_settings = P::make_settings(s.target_id.value());
				desc.add(P::make_options(label("options").c_str(),
				                         s.target_id.value(),
				                         s.target_settings));
			}

			return desc;
		}
	};
};
template<class P>
struct factory {
	using config =
		factory_config<typename P::target,
	                   typename P::targets,
	                   typename detail::var<P, typename P::targets>::settings>;

	using target = typename config::target;
	using targets = typename config::targets;
	using settings_types = typename config::settings_types;
	using optdesc = typename config::optdesc;

	using settings = typename config::settings;
	using options = typename config::template options<factory>;

	static settings_types make_settings(target r) {
		return make_settings(r, targets());
	}

	static optdesc
	make_options(const char * pre, target r, settings_types & s) {
		return make_options(pre, r, s, targets());
	}

	template<class... Args>
	static auto make(const settings & s, Args &&... args) {
		auto var = make_policy(s, std::forward<Args>(args)...);
		return op::core<factory_prod<decltype(var)>>(
			factory_prod<decltype(var)>{std::move(var)});
	}

	template<class... Args>
	static auto make_policy(const settings & s, Args &&... args) {
		return make(s, targets(), std::forward<Args>(args)...);
	}

private:
	template<auto... V, class... Args>
	static auto make(const settings & s, includes<V...>, Args &&... args) {
		using var_t = std::variant<decltype(P::template registry<V>::make(
			typename P::template registry<V>::settings{},
			std::forward<Args>(args)...))...>;
		std::optional<var_t> ret;
		(
			[&](auto v) {
				using type = std::decay_t<decltype(v)>;
				if (s.target_id.value() == v.value) {
					ret.emplace(P::template registry<type::value>::make(
						std::get<typename P::template registry<
							type::value>::settings>(s.target_settings),
						std::forward<Args>(args)...));
				}
			}(flecsi::util::constant<V>{}),
			...);
		return ret.value();
	}

	template<auto... V>
	static settings_types make_settings(target r, includes<V...>) {
		settings_types s;
		(
			[&](auto v) {
				using type = std::decay_t<decltype(v)>;
				if (r == v.value)
					s = typename P::template registry<type::value>::settings{};
			}(flecsi::util::constant<V>{}),
			...);
		return s;
	}

	template<auto... V>
	static optdesc make_options(const char * pre,
	                            target r,
	                            settings_types & s,
	                            includes<V...>) {
		return std::visit(
			[=](auto && arg) {
				optdesc desc;
				(
					[&](auto v) {
						using type = std::decay_t<decltype(v)>;
						if (r == v.value) {
							if constexpr (std::is_same_v<
											  std::decay_t<decltype(arg)>,
											  typename P::template registry<
												  type::value>::settings>) {
								desc.add(typename P::template registry<
										 type::value>::options(pre)(arg));
							}
						}
					}(flecsi::util::constant<V>{}),
					...);
				return desc;
			},
			s);
	}
};

template<class... T>
struct target_union {
	std::variant<T...> value;
};

template<class... T>
std::istream & operator>>(std::istream & in, target_union<T...> & tun) {
	std::tuple<T...> regs;
	bool allbad{true};
	std::string tok;
	in >> tok;
	std::apply(
		[&](auto &... rs) {
			(
				[&](auto & r) {
					std::istringstream myin(tok);
					myin >> r;
					if (!myin.fail()) {
						tun.value = r;
						allbad = false;
					}
				}(rs),
				...);
		},
		regs);

	if (allbad)
		in.setstate(std::ios_base::failbit);
	return in;
}

template<class... Facts>
struct factory_union {
	using config =
		factory_config<target_union<typename Facts::target...>,
	                   std::variant<typename Facts::targets...>,
	                   std::variant<typename Facts::settings_types...>>;

	using optdesc = typename config::optdesc;
	using target = typename config::target;
	using targets = typename config::targets;
	using settings_types = typename config::settings_types;

	using settings = typename config::settings;
	using options = typename config::template options<factory_union>;

	static settings_types make_settings(target r) {
		return std::visit(
			[](auto tval) {
				settings_types ret;
				using TT = std::decay_t<decltype(tval)>;
				(
					[&](auto v) {
						using FT = std::decay_t<decltype(v)>;
						if constexpr (std::is_same_v<TT, typename FT::target>) {
							ret = FT::make_settings(tval);
						}
					}(Facts{}),
					...);
				return ret;
			},
			r.value);
	}

	static optdesc
	make_options(const char * pre, target r, settings_types & s) {
		optdesc ret;
		std::visit(
			[&](auto tval) {
				using TT = std::decay_t<decltype(tval)>;
				return std::visit(
					[=, &ret](auto & sval) {
						using ST = std::decay_t<decltype(sval)>;
						(
							[&](auto v) {
								using FT = std::decay_t<decltype(v)>;
								if constexpr (
									std::is_same_v<TT, typename FT::target> &&
									std::is_same_v<
										ST,
										typename FT::settings_types>) {
									ret.add(FT::make_options(pre, tval, sval));
								}
							}(Facts{}),
							...);
					},
					s);
			},
			r.value);
		return ret;
	}

	template<class... Args>
	static auto make(const settings & s, Args &&... args) {
		auto var = make_policy(s, std::forward<Args>(args)...);
		return op::core<factory_prod<decltype(var)>>(
			factory_prod<decltype(var)>{std::move(var)});
	}

	template<class... Args>
	static auto make_policy(const settings & s, Args &&... args) {
		using var_t = detail::var_cat_t<decltype(Facts::make_policy(
			typename Facts::settings(), std::forward<Args>(args)...))...>;
		std::optional<var_t> ret;
		std::visit(
			[&](auto tval) {
				using TT = std::decay_t<decltype(tval)>;
				(
					[&](auto v) {
						using FT = std::decay_t<decltype(v)>;
						if constexpr (std::is_same_v<TT, typename FT::target>) {
							ret.emplace(FT::make_policy(
								tval, std::forward<Args>(args)...));
						}
					}(Facts{}),
					...);
			},
			s.target_settings);
		return ret.value();
	}
};

}
#endif
