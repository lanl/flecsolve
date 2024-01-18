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

template<class Settings, class Options, class Workgen>
struct krylov_opreg {
	using settings = Settings;
	using options = Options;
	using workgen = Workgen;
	template<class V, class... Args>
	static auto make(const settings & s, const V & v, Args &&... args) {
		return op::krylov(op::krylov_parameters(
			s, workgen::get(v), std::forward<Args>(args)...));
	}
};

template<>
struct krylov_registry<krylov_target::cg>
	: krylov_opreg<cg::settings, cg::options, cg::topo_work<>> {};

template<>
struct krylov_registry<krylov_target::gmres>
	: krylov_opreg<gmres::settings, gmres::options, gmres::topo_work<>> {};

template<>
struct krylov_registry<krylov_target::bicgstab>
	: krylov_opreg<bicgstab::settings,
                   bicgstab::options,
                   bicgstab::topo_work<>> {};

template<>
struct krylov_registry<krylov_target::nka>
	: krylov_opreg<nka::settings, nka::options, nka::topo_work<>> {};

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
