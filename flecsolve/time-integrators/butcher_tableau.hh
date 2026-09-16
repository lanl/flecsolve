/*
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
operated by Triad National Security, LLC for the U.S. Department of
Energy/National Nuclear Security Administration. All rights in the
program are reserved by Triad National Security, LLC, and the
U.S. Department of Energy/National Nuclear Security Administration.
The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material
to reproduce, prepare. derivative works, distribute copies to the
public, perform publicly and display publicly, and to permit others to
do so.
*/
#ifndef FLECSI_LINALG_TIME_INTEGRATOR_BUTCHER_TABLEAU_H
#define FLECSI_LINALG_TIME_INTEGRATOR_BUTCHER_TABLEAU_H

#include <array>
#include <cstddef>

namespace flecsolve::time_integrator {

template<std::size_t Stages>
struct butcher_tableau {
	std::array<double, Stages> c{};
	std::array<std::array<double, Stages>, Stages> a{};
	std::array<double, Stages> b{};
	std::array<double, Stages> b_embedded{};
};

}

#endif
