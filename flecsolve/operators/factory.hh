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

template<class P, template<class> class StoragePolicy = shared_storage>
struct factory {
	using target = typename P::target;
	using targets = typename P::targets;
	template<class T>
	struct var {};
	template<target... V>
	struct var<includes<V...>> {
		using settings =
			std::variant<typename P::template registry<V>::settings...>;
	};
	using settings_types = typename var<targets>::settings;

	using optdesc = boost::program_options::options_description;

	struct settings {
		std::optional<target> target_id;
		settings_types target_settings;
	};
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
				s.target_settings = make_settings(s.target_id.value());
				desc.add(make_options(label("options").c_str(),
				                      s.target_id.value(),
				                      s.target_settings));
			}

			return desc;
		}
	};

	static settings_types make_settings(target r) {
		return make_settings(r, targets());
	}

	static optdesc
	make_options(const char * pre, target r, settings_types & s) {
		return make_options(pre, r, s, targets());
	}
	template<class... Args>
	static auto make(const settings & s, Args &&... args) {
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
		op::core<factory_prod<var_t>, shared_storage> o(
			factory_prod<var_t>{std::move(ret.value())});
		return o;
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

}
#endif
