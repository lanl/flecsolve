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
#ifndef FLECSOLVE_TIME_INTEGRATOR_BDF_COEFFICIENTS_H
#define FLECSOLVE_TIME_INTEGRATOR_BDF_COEFFICIENTS_H

#include <array>
#include <cstddef>

namespace flecsolve::time_integrator::bdf {

struct bdf_coefficients {
	std::array<std::array<double, 6>, 7> source{};
	std::array<double, 7> gamma{};
};

inline constexpr bdf_coefficients constant_step_coefficients{
	// Source coefficients for u_n, u_(n-1), ...
	{{{0., 0., 0., 0., 0., 0.},
	  {-1., 0., 0., 0., 0., 0.},
	  {-2., 0.5, 0., 0., 0., 0.},
	  {-18. / 11., 9. / 11., -2. / 11., 0., 0., 0.},
	  {-48. / 25., 36. / 25., -16. / 25., 3. / 25., 0., 0.},
	  {-300. / 137., 300. / 137., -200. / 137., 75. / 137., -12. / 137., 0.},
	  {-360. / 147., 450. / 147., -400. / 147., 225. / 147., -72. / 147.,
	   10. / 147.}}},

	// Time-operator scaling divided by the timestep.
	{0., 1., 2. / 3., 6. / 11., 12. / 25., 60. / 137., 60. / 147.},
};

}

#endif
