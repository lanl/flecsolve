#ifndef FLECSOLVE_SOLVERS_MG_LEVEL_HH
#define FLECSOLVE_SOLVERS_MG_LEVEL_HH

#include <tuple>
#include <optional>

#include "flecsolve/operators/core.hh"
#include "flecsolve/operators/handle.hh"

namespace flecsolve::mg {

enum class oplabel : std::size_t { A, presmoother, postsmoother, P, R, size };
enum class veclabel : std::size_t { rhs, sol, res, correction, size };


template<class P, class R, class... Ops>
struct tuple_opstore {

	template<class ... Args,
	         std::enable_if_t<sizeof...(Args) == static_cast<std::size_t>(oplabel::size), bool> = true>
	tuple_opstore(Args && ... args) :
		ops{std::forward<Args>(args)...} {
		// static_assert((... && op::is_operator_v<std::decay_t<Args>>));
	}

	template<class ... Args,
	         std::enable_if_t<sizeof...(Args) != static_cast<std::size_t>(oplabel::size), bool> = true>
	tuple_opstore(Args && ... args) :
		ops{std::forward<Args>(args)..., {}, {}} {
		static_assert(sizeof...(Args) == static_cast<std::size_t>(oplabel::size) - 2);
		// static_assert((... && op::is_operator_v<std::decay_t<Args>>));
	}

	constexpr std::size_t size() const {
		return sizeof...(Ops);
	}

	template<oplabel L>
	auto & get () {
		if constexpr (static_cast<std::size_t>(L) >=
		              static_cast<std::size_t>(oplabel::size) - 2)
			return std::get<static_cast<std::size_t>(L)>(ops).value().get();
		else
			return std::get<static_cast<std::size_t>(L)>(ops).get();
	}
	template<oplabel L>
	const auto & get () const {
		if constexpr (static_cast<std::size_t>(L) >=
		              static_cast<std::size_t>(oplabel::size) - 2)
			return std::get<static_cast<std::size_t>(L)>(ops).value().get();
		else
			return std::get<static_cast<std::size_t>(L)>(ops).get();
	}

	std::tuple<op::handle<Ops>...,
	           std::optional<op::handle<P>>,
	           std::optional<op::handle<R>>> ops;

};

template<template<class ...> class OpStorage,
         template<class> class VecStorage,
         class ... Ops>
struct level :
		OpStorage<Ops...>,
    	VecStorage<std::tuple_element_t<2, std::tuple<std::decay_t<Ops>...>>> {
	using opstore = OpStorage<Ops...>;
	using vecstore = VecStorage<std::tuple_element_t<2, std::tuple<std::decay_t<Ops>...>>>;

	template<class ... Args>
	level(Args && ... args) :
		opstore(std::forward<Args>(args)...),
		vecstore(opstore::template get<oplabel::A>())
	{}

	auto & A() { return opstore::template get<oplabel::A>(); }
	const auto & A() const { return opstore::template get<oplabel::A>(); }

	auto & P() { return opstore::template get<oplabel::P>(); }
	const auto & P() const { return opstore::template get<oplabel::P>(); }

	auto & R() { return opstore::template get<oplabel::R>(); }
	const auto & R() const { return opstore::template get<oplabel::R>(); }

	auto & presmoother() { return opstore::template get<oplabel::presmoother>(); }
	const auto & presmoother() const { return opstore::template get<oplabel::presmoother>(); }

	auto & postsmoother() { return opstore::template get<oplabel::postsmoother>(); }
	const auto & postsmoother() const { return opstore::template get<oplabel::postsmoother>(); }

	auto & rhs() { return vecstore::template get<veclabel::rhs>(); }
	const auto & rhs() const { return vecstore::template get<veclabel::rhs>(); }

	auto & sol() { return vecstore::template get<veclabel::sol>(); }
	const auto & sol() const { return vecstore::template get<veclabel::sol>(); }

	auto & res() { return vecstore::template get<veclabel::res>(); }
	const auto & res() const { return vecstore::template get<veclabel::res>(); }

	auto & correction() { return vecstore::template get<veclabel::correction>(); }
	const auto & correction() const { return vecstore::template get<veclabel::correction>(); }
};


template<class LevelPolicy>
struct hierarchy : LevelPolicy
{
	using level_type = typename LevelPolicy::level_type;
	level_type & get(int level) {
		if (level < 0) level = levels.size() + level;
		return levels[level];
	}

	const level_type & get(int level) const {
		if (level < 0) level = levels.size() + level;
		return levels[level];
	}

	decltype(auto) get_mat(int level) {
		return levels[level].A();
	}

	template<class ... O>
	hierarchy(O && ... o) {
		extend(std::forward<O>(o)...);
	}

	template<class ... O>
	void extend(O && ... o) {
		levels.emplace_back(
			std::forward<O>(o)...);
	}

	std::size_t depth() const { return levels.size(); }

private:
	std::vector<level_type> levels;
};
}

#endif
