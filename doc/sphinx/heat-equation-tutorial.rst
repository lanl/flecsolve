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

The explicit driver uses ``integrator_factory`` to select RK23 or RK45 from
the ``type`` setting in ``explicit.cfg``. Method settings belong in the
``[time-integrator.options]`` section. The factory uses the heat operator
and allocates work vectors from the solution vector:

.. literalinclude:: ../../examples/heat_equation/explicit.cc
   :language: cpp
   :lines: 8-52

At each iteration, ``advance(dt, u, unew)`` writes a candidate solution into
``unew`` without advancing the integrator's clock. If ``check_solution()``
accepts the candidate, the driver calls ``update()`` to commit time and
integrator state, then swaps ``u`` and ``unew`` to accept the solution.

If the candidate is rejected, it obtains a retry timestep from
``get_next_dt(false)`` and calls ``advance`` again from the same
solution and time. After an accepted step, ``get_next_dt(true)``
supplies the next timestep.

Given only an RHS handle, ``integrator_factory`` builds RK methods; selecting
``type = bdf`` this way throws ``std::invalid_argument``, because BDF also
needs a time operator and a solver. The implicit driver below shows that form.

Applications that manage their own clock and timesteps can use
``time_integrator::stepper_factory``. Its configuration requires only
``type = rk23`` or ``type = rk45`` in the selected section for RK methods. Construction
uses the same prototype vector and RHS handle as ``integrator_factory``::

   auto stepper = stepper_factory::make(
       read_config("stepper.cfg", stepper_factory::options("stepper")),
       u, op::ref(F));
   stepper.advance(dt, u, unew);
   std::swap(u, unew);

Here ``stepper.cfg`` contains a ``[stepper]`` section with ``type = rk45``;
``flecsolve/time-integrators/test/stepper-factory.cfg`` has complete RK and
BDF sections. ``dt`` may be a scalar or a FleCSI future. The application chooses each
step and advances its own clock. To select the method in code, use
``stepper_factory::make(stepper_factory::target::rk45, u, op::ref(F))``.

The same factory also selects BDF with ``type = bdf`` and method settings
in ``[stepper.options]``. For that path, supply the implicit time operator
and solver with ``stepper_factory::make(settings, u, F, solver)``. After
``advance()``, call ``check_solution()`` and commit history with ``update()``
only on acceptance, then swap vectors and advance the application clock.
Retain the solution and time on rejection and choose a retry timestep.
BDF requires a scalar timestep. The solver-taking overload can also select
RK, in which case it uses ``F.apply_rhs()`` and does not invoke the solver.

Implicit integration
--------------------

The implicit driver uses ``integrator_factory`` with a time operator and a
Krylov solver.
An ``operator_adapter`` turns the heat operator ``F = alpha Laplacian`` into the operator required by
the implicit method, such as ``I - gamma F``. The solver factory selects the
linear solver from ``implicit.cfg``:

.. literalinclude:: ../../examples/heat_equation/implicit.cc
   :language: cpp
   :lines: 8-53

``[time-integrator]`` selects ``type = bdf``. The existing BDF options live
in ``[time-integrator.options]``: ``integrator`` selects BE, CN, or BDF2--6,
and ``starting-integrator`` selects BE or CN for startup. The application
constructs the solver separately and passes it to ``integrator_factory::make``
along with the prototype vector and time operator. The integrator and solver
must share operator state so that BDF's scaling updates reach the solver.
Because this overload accepts every method, setting ``type = rk45`` (with RK
options) in ``implicit.cfg`` runs the same driver explicitly: RK reads the
RHS through ``F.apply_rhs()`` and never calls the solver.

The factory exposes ``advance(dt, current, next)`` and the same acceptance
loop as the explicit driver. BDF stays in startup until a step is accepted,
so a rejected first attempt is retried as a startup step.
Call ``update()`` and swap solution vectors only after acceptance.

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
