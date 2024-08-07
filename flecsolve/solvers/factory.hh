/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the U.S.
Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others
to do so.
*/
#ifndef FLECSOLVE_SOLVER_FACTORY_H
#define FLECSOLVE_SOLVER_FACTORY_H

#include <string>

#include "flecsolve/operators/factory.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/gmres.hh"
#include "flecsolve/solvers/bicgstab.hh"
#include "flecsolve/solvers/nka.hh"

namespace flecsolve {
enum class krylov_target { cg, gmres, bicgstab, nka };
template<krylov_target T>
struct krylov_registry {};

template<template<class> class Solver, class Settings, class Options, class Workgen>
struct krylov_opreg {
	using settings = Settings;
	using options = Options;
	using workgen = Workgen;
	template<class V, class... Args>
	static auto make(const settings & s, const V & v, Args &&... args) {
		return Solver(s, workgen{}(v))(std::forward<Args>(args)...);
	}
};

template<>
struct krylov_registry<krylov_target::cg>
	: krylov_opreg<cg::solver, cg::settings, cg::options, decltype(cg::make_work)> {};

template<>
struct krylov_registry<krylov_target::gmres>
	: krylov_opreg<gmres::solver, gmres::settings, gmres::options, decltype(gmres::make_work)> {};

template<>
struct krylov_registry<krylov_target::bicgstab>
	: krylov_opreg<bicgstab::solver, bicgstab::settings,
                   bicgstab::options,
	               decltype(bicgstab::make_work)> {};

template<>
struct krylov_registry<krylov_target::nka>
	: krylov_opreg<nka::solver, nka::settings, nka::options,
	               decltype(nka::make_work)> {};

inline std::istream & operator>>(std::istream & in, krylov_target & reg) {
	std::string tok;
	in >> tok;

	if (tok == "cg")
		reg = krylov_target::cg;
	else if (tok == "gmres")
		reg = krylov_target::gmres;
	else if (tok == "bicgstab")
		reg = krylov_target::bicgstab;
	else if (tok == "nka")
		reg = krylov_target::nka;
	else
		in.setstate(std::ios_base::failbit);

	return in;
}

struct krylov_factory_policy {
	using target = krylov_target;
	using targets = flecsi::util::
		constants<target::cg, target::gmres, target::bicgstab, target::nka>;

	template<target V>
	using registry = krylov_registry<V>;
};

using krylov_factory = op::factory<krylov_factory_policy>;

}

#endif
