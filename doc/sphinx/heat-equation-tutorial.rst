Heat Equation Tutorial
======================

This tutorial walks through the two-dimensional heat-equation example in
``examples/heat_equation``. It shows how a FleCSI mesh and fields become
flecsolve vectors, how the discrete Laplacian is exposed as an operator,
and how explicit and implicit time integrators use that operator.

The complete, buildable source for this tutorial is in the repository under
``examples/heat_equation``. The snippets below are included directly from
that source so that the tutorial stays aligned with the example.

Problem
-------

The example solves the heat equation

.. math::

   \frac{\partial u}{\partial t} = \alpha \Delta u
   \quad \text{in } \Omega = (0,10) \times (0,10),

with homogeneous Dirichlet boundary conditions and a hot square as the
initial condition:

.. math::

   u = 0 \text{ on } \partial\Omega, \qquad
   u(x,y,0) = \begin{cases}
     50 & 4 \leq x \leq 6,\ 4 \leq y \leq 6, \\
     0 & \text{otherwise.}
   \end{cases}

The mesh is distributed across flecsi colors. The command-line mesh
extents specify the number of points in the ``x`` and ``y``
directions.

Build the example
-----------------

Configure the project with examples enabled and build it:

.. code-block:: console

   cmake -S . -B build -DFLECSOLVE_BUILD_EXAMPLES=ON
   cmake --build build

The executables and their configuration files are written to
``build/examples/heat_equation``.

Run the example
---------------

Run the explicit RK23 version on four MPI processes:

.. code-block:: console

   cd build/examples/heat_equation
   mpirun -np 4 ./heat-explicit 100 100 -d 1.5 -o true

Run the implicit BDF version with the same mesh and diffusivity:

.. code-block:: console

   mpirun -np 4 ./heat-implicit 100 100 -d 1.5 -o true

The positional arguments are the ``x`` and ``y`` mesh extents. The ``-d``
option sets the diffusivity ``alpha`` and ``-o true`` writes every accepted
time step. The ``explicit.cfg`` and ``implicit.cfg`` files control the time
integrator and linear-solver settings.

The final output is written by the ``finalize`` control point. With
``-o true``, intermediate files are also written using names such as
``timestep-0-0.dat``. The files contain ``x``, ``y``, and ``u`` columns and
can be visualized with the utilities in ``examples/heat_equation/util``.

Mesh and vectors
----------------

The example specializes FleCSI's ``narray`` topology as a two-dimensional
mesh. Two fields hold the current and next solution values. The control
policy turns those fields into flecsolve topology views after the mesh has
been allocated:

.. literalinclude:: ../../examples/heat_equation/control.hh
   :language: cpp
   :lines: 39-66

The topology and field definitions are in
``examples/heat_equation/mesh.hh``. A topology view lets the time integrator
operate on FleCSI fields through the standard flecsolve vector interface.

Initial and boundary conditions
-------------------------------

The ``initialize`` control point first allocates the mesh and then launches
the ``ics`` task. That task sets the value to ``50`` inside the square
``[4, 6] x [4, 6]`` and to zero elsewhere. The implementation is in
``examples/heat_equation/heat.cc``.

The discrete Laplacian applies the zero Dirichlet boundary condition before
computing interior values. The boundary checks account for MPI subdomains,
so only processes that own a global boundary write those boundary values.

Discrete operator
-----------------

The ``laplace`` task uses the mesh spacing and a centered finite-difference
stencil. It computes ``alpha * Laplacian(u)`` into the output vector:

.. literalinclude:: ../../examples/heat_equation/heat.hh
   :language: cpp
   :lines: 65-115

The task is wrapped in ``heat_op``, which implements the flecsolve operator
interface. Its ``apply`` method launches the task with the topology and
field references obtained from the input and output vectors:

.. literalinclude:: ../../examples/heat_equation/heat.hh
   :language: cpp
   :lines: 119-131

Explicit integration
--------------------

The explicit driver constructs an RK23 integrator using the heat operator,
the settings read from ``explicit.cfg``, and work vectors created from the
solution vector:

.. literalinclude:: ../../examples/heat_equation/explicit.cc
   :language: cpp
   :lines: 8-51

At each iteration, ``advance`` proposes a new solution. The driver checks
the result, updates the integrator state, swaps the current and next vectors,
and asks the integrator for the next time step.

Implicit integration
--------------------

The implicit driver uses BDF and a Krylov solver. An ``operator_adapter``
turns the heat operator ``F = alpha Laplacian`` into the operator required by
the implicit method, such as ``I - gamma F``. The solver factory selects the
linear solver from ``implicit.cfg``:

.. literalinclude:: ../../examples/heat_equation/implicit.cc
   :language: cpp
   :lines: 8-57

Next steps
----------

To experiment with the example, try changing the following:

* Set ``diffusivity`` with ``-d`` to change the rate of diffusion.
* Change ``initial-dt``, ``max-dt``, or the final time in the configuration
  files.
* Increase the mesh extents to study spatial resolution and parallel scaling.
* Modify ``task::ics`` in ``heat.cc`` to use a different initial condition.
* Modify ``task::laplace`` in ``heat.hh`` to experiment with another stencil
  or boundary condition.
