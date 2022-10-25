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

namespace flecsolve {

namespace po = boost::program_options;

struct with_solver_erasure {
	struct solver_parameters {
		virtual ~solver_parameters() {}

		virtual po::options_description options() = 0;
	};
	struct solver {
		virtual ~solver() {}
		virtual solve_info apply(std::any x, std::any y) = 0;
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
						f(reg);
					}),
				"solver type");
		}

		if (parameters) {
			desc.add(parameters->options());
		}

		return desc;
	}

	std::shared_ptr<solver_parameters> get_parameters() { return parameters; }

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
	template<class D, class R, class S>
	struct solver_wrapper : solver {
		solver_wrapper(S & s) : slv(s) {}

		solve_info apply(std::any x, std::any y) override {
			return slv.apply(*(std::any_cast<D *>(x)),
			                 *(std::any_cast<R *>(y)));
		}

	protected:
		S & slv;
	};

	std::shared_ptr<solver_parameters> parameters;
};

struct krylov_factory : solver_factory<krylov_factory> {
	enum class registry { cg, gmres, bicgstab };

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

	void set_solver_type(registry reg) { solver_type = reg; }

	template<class V, class... Args>
	void create_parameters(registry reg, V & v, Args &&... args) {
		if (!parameters) {
			switch (reg) {
				case registry::cg: {
					parameters = make_params<con_types<registry::cg>>(
						v, std::forward<Args>(args)...);
					break;
				}
				case registry::gmres: {
					parameters = make_params<con_types<registry::gmres>>(
						v, std::forward<Args>(args)...);
					break;
				}
				case registry::bicgstab: {
					parameters = make_params<con_types<registry::bicgstab>>(
						v, std::forward<Args>(args)...);
				}
			}
		}
	}

	template<class V, class... Args>
	std::unique_ptr<storage> create(V & v, Args &&... args) {
		std::unique_ptr<storage> ret;
		assert(parameters);

		switch (solver_type) {
			case registry::cg: {
				ret = make_storage<con_types<registry::cg>>(
					v, std::forward<Args>(args)...);
				break;
			}
			case registry::gmres: {
				ret = make_storage<con_types<registry::gmres>>(
					v, std::forward<Args>(args)...);
				break;
			}
			case registry::bicgstab: {
				ret = make_storage<con_types<registry::bicgstab>>(
					v, std::forward<Args>(args)...);
			}
		}

		return ret;
	}

	template<class D, class R, class V, class... Args>
	std::unique_ptr<solver> wrap(storage * store, V & v, Args &&... args) {
		std::unique_ptr<solver> ret;

		switch (solver_type) {
			case registry::cg: {
				ret = make_wrapper<con_types<registry::cg>, D, R>(
					store, v, std::forward<Args>(args)...);
				break;
			}
			case registry::gmres: {
				ret = make_wrapper<con_types<registry::gmres>, D, R>(
					store, v, std::forward<Args>(args)...);
				break;
			}
			case registry::bicgstab: {
				ret = make_wrapper<con_types<registry::bicgstab>, D, R>(
					store, v, std::forward<Args>(args)...);
			}
		}

		return ret;
	}

protected:
	template<class ctypes, class V, class... Args>
	auto make_params(V & v, Args &&... args) {
		op::krylov_parameters params(
			typename ctypes::settings(label("parameters").c_str()),
			ctypes::workgen::get(v),
			std::forward<Args>(args)...);
		return std::make_shared<solver_param_wrapper<decltype(params)>>(
			std::move(params));
	}

	template<class ctypes, class V, class... Args>
	auto make_storage(V & v, Args &&... args) {
		auto * params =
			dynamic_cast<solver_param_wrapper<decltype(op::krylov_parameters(
				typename ctypes::settings(""),
				ctypes::workgen::get(v),
				std::forward<Args>(args)...))> *>(parameters.get());
		assert(params);

		// rebind operators in case they were moved
		auto & old_params = params->target;
		op::krylov_parameters new_params(std::move(old_params.solver_settings),
		                                 ctypes::workgen::get(v),
		                                 std::forward<Args>(args)...);
		parameters =
			std::make_shared<solver_param_wrapper<decltype(new_params)>>(
				std::move(new_params));

		auto * new_params1 =
			dynamic_cast<solver_param_wrapper<decltype(op::krylov_parameters(
				typename ctypes::settings(""),
				ctypes::workgen::get(v),
				std::forward<Args>(args)...))> *>(parameters.get());
		assert(new_params1);
		op::krylov slv(new_params1->target);

		return std::make_unique<storage_wrapper<decltype(slv)>>(std::move(slv));
	}

	template<class ctypes, class D, class R, class V, class... Args>
	auto make_wrapper(storage * store, V & v, Args &&... args) {
		auto * params =
			dynamic_cast<solver_param_wrapper<decltype(op::krylov_parameters(
				typename ctypes::settings(""),
				ctypes::workgen::get(v),
				std::forward<Args>(args)...))> *>(parameters.get());
		assert(params);

		auto * slv = dynamic_cast<
			storage_wrapper<decltype(op::krylov(params->target))> *>(store);
		assert(slv);

		return std::make_unique<solver_wrapper<
			D,
			R,
			typename std::remove_reference_t<decltype(*slv)>::type>>(
			slv->target);
	}

	registry solver_type;
};

inline std::istream & operator>>(std::istream & in,
                                 krylov_factory::registry & reg) {
	std::string tok;
	in >> tok;

	if (tok == "cg")
		reg = krylov_factory::registry::cg;
	else if (tok == "gmres")
		reg = krylov_factory::registry::gmres;
	else if (tok == "bicgstab")
		reg = krylov_factory::registry::bicgstab;
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
	std::unique_ptr<storage> create(V & v, Args &&... args) {
		std::unique_ptr<storage> ret;
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
									ret = fact.create(
										v, std::forward<Args>(args)...);
								}
							}(facts),
							...);
					},
					members);
			},
			solver_type.value);

		return ret;
	}

	template<class D, class R, class V, class... Args>
	std::unique_ptr<solver> wrap(storage * store, V & v, Args &&... args) {
		std::unique_ptr<solver> ret;

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
									ret = fact.template wrap<D, R>(
										store, v, std::forward<Args>(args)...);
								}
							}(facts),
							...);
					},
					members);
			},
			solver_type.value);

		return ret;
	}

protected:
	registry solver_type;
	std::tuple<Facts...> members;
};

}

#endif
