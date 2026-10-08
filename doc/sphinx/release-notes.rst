Release Notes
=============

1.0.0
-----

Initial release.

Added
~~~~~

* FleCSI-backed vector interface adapters and multi-component vector
  support.
* Operator interfaces and ownership-aware operator handles.
* Sequential and parallel CSR matrix support.
* Krylov solver components including conjugate gradient, BiCGSTAB,
  GMRES, and nonlinear Krylov acceleration.
* Multigrid components and AMP-backed solver wrappers.
* Explicit and implicit time integration components.
* Example applications for heat equation, Poisson, and equilibrium
  diffusion workflows.
* Spack package definitions for flecsolve.
* Initial Sphinx documentation and documentation build workflow for
  GitHub Pages publishing.

1.0.1
-----

Added
~~~~~

* Standalone RK23, RK45, and BDF steppers for applications that manage
  their own time steps, without the controlled integrator interface.
* Future-valued scalar arguments for vector initialization, scaling,
  scalar addition, and linear combinations, including mixed scalar/future
  coefficients.
* Future-valued time steps for the RK23 and RK45 standalone steppers.

Changed
~~~~~~~

* Refactored time integrators to separate stepping from time bookkeeping
  and step-size selection. Existing controlled integrators remain available;
  the BDF stepper retains solution history and acceptance checks.
* Deferred scalar transforms on futures until the consuming vector task
  executes, including RK stage coefficients derived from future time steps.
* Documented future-valued vector operations and standalone stepper usage
  in the :doc:`components` guide.
* Removed Fortran in CMake.
