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

Unreleased
----------

Added
~~~~~

* ``time_integrator::integrator_factory`` and
  ``time_integrator::stepper_factory`` in ``time-integrators/factory.hh``
  select RK23, RK45, or BDF from a configuration file. Construction from an
  RHS handle supports RK methods; construction from a time operator and a
  solver supports every method, with RK reading the RHS through
  ``apply_rhs()``.
* ``num_step_rejects()`` on the RK23 and RK45 integrators.
* A ``flecsolve-doc`` CMake target that builds the Sphinx documentation
  when ``sphinx-build`` is available.

Changed
~~~~~~~

* ``bdf::stepper::advance`` and ``bdf::integrator::advance`` no longer take
  a ``first_step`` flag; call ``advance(dt, current, next)``. BDF now stays
  in startup until the first step is accepted, which the flag could
  previously contradict.
* The heat equation examples construct their integrators through
  ``integrator_factory``. Their configuration files now select the method
  with ``type`` in ``[time-integrator]`` and hold method settings in
  ``[time-integrator.options]``; configurations written for the previous
  examples must be updated to this layout.

Fixed
~~~~~

* ``bdf::stepper`` no longer divides by a zero previous timestep on its
  first step, and a retry after a rejected first attempt no longer uses the
  rejected timestep in its startup error estimate.
* BDF steppers and integrators can be moved between operations without
  their solution history referring to the moved-from work vectors.
* Several ``bdf::stepper`` members are now initialized before first use.
