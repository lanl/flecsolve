#ifndef FLECSOLVE_SOLVER_FACTORY_H
#define FLECSOLVE_SOLVER_FACTORY_H

#include <string>
#include <variant>
#include <any>
#include <boost/program_options/options_description.hpp>

#include "flecsolve/util/config.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/gmres.hh"
#include "flecsolve/solvers/bicgstab.hh"
#include "flecsolve/solvers/nka.hh"

namespace flecsolve {

namespace po = boost::program_options;

struct with_solver_erasure {
	struct solver_parameters {
		virtual ~solver_parameters() {}

		virtual po::options_description options() = 0;
	};
	struct storage {
		virtual ~storage() {}
	};
};

template<class P>
struct solver_factory : with_solver_erasure, with_label {
	solver_factory() : with_label("") {}

	void set_options_name(const std::string & name) {
		set_prefix(name.c_str());
	}

	template<class F>
	auto options(F f) {
		po::options_description desc;

		if (prefix != "") {
			desc.add_options()(
				label("type").c_str(),
				po::value<typename P::registry>()->required()->notifier(
					[f, this](const typename P::registry & reg) {
						static_cast<P &>(*this).set_solver_type(reg);
						static_cast<P &>(*this).create_parameters(reg);
						f(reg);
					}),
				"solver type");
		}

		if (parameters) {
			desc.add(parameters->options());
		}

		return desc;
	}

	auto options() {
		return options([](const auto &) {});
	}

	std::shared_ptr<solver_parameters> get_parameters() { return parameters; }
	bool has_solver() const { return solver_storage != nullptr; }

protected:
	template<class T>
	struct solver_param_wrapper : solver_parameters {
		solver_param_wrapper(T && t) : target(std::move(t)) {}

		po::options_description options() override { return target.options(); }

		T target;
	};
	template<class T>
	struct storage_wrapper : storage {
		using type = T;
		storage_wrapper(T && t) : target(std::move(t)) {}
		T target;
	};

	std::shared_ptr<solver_parameters> parameters;
	std::shared_ptr<storage> solver_storage;
};

enum class krylov_registry { cg, gmres, bicgstab, nka };
template<class... Ops>
struct krylov_factory : solver_factory<krylov_factory<Ops...>> {

	using base = solver_factory<krylov_factory<Ops...>>;
	using storage = typename base::storage;
	using solver_parameters = typename base::solver_parameters;

	template<class T>
	using solver_param_wrapper =
		typename base::template solver_param_wrapper<T>;
	template<class T>
	using storage_wrapper = typename base::template storage_wrapper<T>;
	using base::label;
	using base::parameters;
	using base::solver_storage;

	using registry = krylov_registry;

	krylov_factory(Ops &&... o) : ops{std::forward<Ops>(o)...} {}

	template<registry v>
	struct con_types {};

	template<>
	struct con_types<registry::cg> {
		using settings = cg::settings;
		using workgen = cg::topo_work<>;
	};

	template<>
	struct con_types<registry::gmres> {
		using settings = gmres::settings;
		using workgen = gmres::topo_work<>;
	};

	template<>
	struct con_types<registry::bicgstab> {
		using settings = bicgstab::settings;
		using workgen = bicgstab::topo_work<>;
	};

	template<>
	struct con_types<registry::nka> {
		using settings = nka::settings;
		using workgen = nka::topo_work<>;
	};
	void set_solver_type(registry reg) { solver_type = reg; }

	void create_parameters(registry reg) {
		if (!parameters) {
			switch (reg) {
				case registry::cg: {
					parameters = make_params<con_types<registry::cg>>();
					break;
				}
				case registry::gmres: {
					parameters = make_params<con_types<registry::gmres>>();
					break;
				}
				case registry::bicgstab: {
					parameters = make_params<con_types<registry::bicgstab>>();
					break;
				}
				case registry::nka: {
					parameters = make_params<con_types<registry::nka>>();
				}
			}
		}
	}

	template<class V, class Op>
	void create(vec::base<V> & v, op::base<Op> & A) {
		assert(parameters);

		if (!solver_storage) {
			switch (solver_type) {
				case registry::cg: {
					make_storage<con_types<registry::cg>>(
						v.derived(), make_is(), A.derived());
					break;
				}
				case registry::gmres: {
					make_storage<con_types<registry::gmres>>(
						v.derived(), make_is(), A.derived());
					break;
				}
				case registry::bicgstab: {
					make_storage<con_types<registry::bicgstab>>(
						v.derived(), make_is(), A.derived());
					break;
				}
				case registry::nka: {
					make_storage<con_types<registry::nka>>(
						v.derived(), make_is(), A.derived());
				}
			}
		}
	}

	template<class V, class D, class R, class Op>
	decltype(auto) solve(const vec::base<D> & x,
	                     vec::base<R> & y,
	                     vec::base<V> & v,
	                     op::base<Op> & A) {
		assert(solver_storage);

		switch (solver_type) {
			case registry::cg: {
				return solve_impl<con_types<registry::cg>>(x.derived(),
				                                           y.derived(),
				                                           v.derived(),
				                                           make_is(),
				                                           A.derived());
				break;
			}
			case registry::gmres: {
				return solve_impl<con_types<registry::gmres>>(x.derived(),
				                                              y.derived(),
				                                              v.derived(),
				                                              make_is(),
				                                              A.derived());
				break;
			}
			case registry::bicgstab: {
				return solve_impl<con_types<registry::bicgstab>>(x.derived(),
				                                                 y.derived(),
				                                                 v.derived(),
				                                                 make_is(),
				                                                 A.derived());
				break;
			}
			case registry::nka: {
				return solve_impl<con_types<registry::nka>>(x.derived(),
				                                            y.derived(),
				                                            v.derived(),
				                                            make_is(),
				                                            A.derived());
			}
		}
	}

protected:
	template<class ctypes>
	auto make_params() {
		typename ctypes::settings params(label("parameters").c_str());
		return std::make_shared<solver_param_wrapper<decltype(params)>>(
			std::move(params));
	}

	template<class ctypes, class V, std::size_t... I, class Op>
	void make_storage(V & v, std::index_sequence<I...>, Op & A) {
		auto * params =
			dynamic_cast<solver_param_wrapper<typename ctypes::settings> *>(
				parameters.get());
		assert(params);

		op::krylov slv(
			op::krylov_parameters(params->target,
		                          ctypes::workgen::get(v),
		                          std::ref(A),
		                          std::forward<Ops>(std::get<I>(ops))...));
		solver_storage =
			std::make_shared<storage_wrapper<decltype(slv)>>(std::move(slv));
	}

	template<class ctypes,
	         class D,
	         class R,
	         class V,
	         std::size_t... I,
	         class Op>
	decltype(auto)
	solve_impl(D & x, R & y, V & v, std::index_sequence<I...>, Op & A) {
		typename ctypes::settings settings("");
		auto * slv = dynamic_cast<storage_wrapper<decltype(op::krylov(
			op::krylov_parameters(settings,
		                          ctypes::workgen::get(v),
		                          std::ref(A),
		                          std::forward<Ops>(std::get<I>(ops))...)))> *>(
			solver_storage.get());
		assert(slv);

		return slv->target.apply(x, y);
	}

	registry solver_type;
	std::tuple<std::decay_t<Ops>...> ops;
	using make_is = std::make_index_sequence<sizeof...(Ops)>;
};
template<class... O>
krylov_factory(O &&...) -> krylov_factory<O...>;

inline std::istream & operator>>(std::istream & in, krylov_registry & reg) {
	std::string tok;
	in >> tok;

	if (tok == "cg")
		reg = krylov_registry::cg;
	else if (tok == "gmres")
		reg = krylov_registry::gmres;
	else if (tok == "bicgstab")
		reg = krylov_registry::bicgstab;
	else if (tok == "nka")
		reg = krylov_registry::nka;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

namespace detail {
template<class... R>
struct union_registry {
	std::variant<R...> value;
};

template<class... R>
std::istream & operator>>(std::istream & in, union_registry<R...> & ureg) {
	std::tuple<R...> regs;
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
						ureg.value = r;
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
}

template<class... Facts>
struct factory_union : solver_factory<factory_union<Facts...>> {
	using base = solver_factory<factory_union<Facts...>>;
	using registry = detail::union_registry<typename Facts::registry...>;
	using storage = typename base::storage;
	using solver = typename base::solver;
	using base::parameters;

	void set_options_name(const std::string & name) {
		base::set_prefix(name.c_str());
		std::apply(
			[&](auto &... fact) { (fact.set_options_name(name.c_str()), ...); },
			members);
	}

	void set_solver_type(registry reg) { solver_type.value = reg.value; }

	template<class V, class... Args>
	void create_parameters(registry reg, V & v, Args &&... args) {
		std::visit(
			[&](auto reg_value) {
				std::apply(
					[&](auto &... facts) {
						(
							[&](auto & fact) {
								using RT = std::decay_t<decltype(reg_value)>;
								if constexpr (std::is_same_v<
												  RT,
												  typename std::decay_t<
													  decltype(fact)>::
													  registry>) {
									fact.create_parameters(
										reg_value,
										v,
										std::forward<Args>(args)...);
									parameters = fact.get_parameters();
								}
							}(facts),
							...);
					},
					members);
			},
			reg.value);
	}

	template<class V, class... Args>
	void create(V & v, Args &&... args) {
		std::visit(
			[&](auto reg_value) {
				std::apply(
					[&](auto &... facts) {
						(
							[&](auto & fact) {
								using RT = std::decay_t<decltype(reg_value)>;
								if constexpr (std::is_same_v<
												  RT,
												  typename std::decay_t<
													  decltype(fact)>::
													  registry>) {
									fact.create(v, std::forward<Args>(args)...);
								}
							}(facts),
							...);
					},
					members);
			},
			solver_type.value);
	}

	template<class D, class R, class V, class... Args>
	decltype(auto) solve(D & x, R & y, V & v, Args &&... args) {
		std::visit(
			[&](auto reg_value) {
				std::apply(
					[&](auto &... facts) {
						(
							[&](auto & fact) {
								using RT = std::decay_t<decltype(reg_value)>;
								if constexpr (std::is_same_v<
												  RT,
												  typename std::decay_t<
													  decltype(fact)>::
													  registry>) {
									return fact.solve(
										x, y, v, std::forward<Args>(args)...);
								}
							}(facts),
							...);
					},
					members);
			},
			solver_type.value);
	}

protected:
	registry solver_type;
	std::tuple<Facts...> members;
};

}

#endif
