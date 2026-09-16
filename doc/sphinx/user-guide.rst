User Guide
==========

This guide summarizes the main programming concepts used by flecsolve.
The interfaces are C++ templates, so application code normally includes
the relevant headers and lets solver, vector, and operator types be
deduced from the concrete FleCSI fields and topology views passed in.

The :doc:`components` guide provides a detailed tour of the vector,
operator, matrix, solver, time-integrator, and physics layers. This page
collects the cross-cutting patterns that are most important when assembling
an application.

Vector Interface
----------------

flecsolve algorithms expect vector-like objects that provide common
linear algebra operations. For FleCSI applications, ``vec::topo_view`` is
the primary adapter from FleCSI fields and topologies into that vector
interface.

Multi-component physics state can be represented with ``vec::multi``.
This is useful when different operators own different physics variables
but a solver or time integrator needs to carry the coupled state as one
object.

Operator Interface
------------------

An operator maps a domain vector to a range vector by implementing an
``apply`` operation. The concrete map may launch FleCSI tasks, call into
matrix kernels, or compose lower-level vector operations.

When passing operators to solvers or time integrators, use operator
handles to make ownership explicit. Shared ownership uses
``op::make_shared``; non-owning references use ``op::ref`` or
``op::cref``.

Krylov Diagnostics
------------------

Krylov solvers can accept an optional diagnostic callback. The callback
can monitor convergence and request early termination.

.. code-block:: cpp

   bool diagnostic(const Vector & current_solution, double residual_norm);

The callback returns ``true`` when the solve should stop early.

Configuration Files
-------------------

Many solver and time-integrator settings are exposed through Boost
program-options wrappers. Existing tests and examples include ``.cfg``
files that show the expected option names and values.

Useful starting points in the repository:

* ``flecsolve/solvers/test/*.cfg``
* ``flecsolve/time-integrators/test/*.cfg``
* ``examples/heat_equation/*.cfg``
* ``examples/poisson/poisson.cfg``
