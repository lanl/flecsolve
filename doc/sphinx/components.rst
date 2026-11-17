Library Components
==================

flecsolve is organized as a set of composable numerical components
rather than as one application framework.

Most applications begin with a physical state stored in FleCSI fields. The
vector layer presents that state to numerical algorithms without requiring
the algorithms to know how the fields are distributed. Operators then define
the physics or algebraic action on those vectors. A matrix, a FleCSI task, or
another application-defined implementation can provide that action as long
as it follows the operator interface.

Solvers and time integrators are consumers of these shared interfaces. A
solver uses an operator to compute residuals and iteratively improve a
solution to a linear or nonlinear problem. A time integrator uses an
operator as the right-hand side of an evolution equation and manages the
work vectors and (for implicit methods) the linear solve required at each
step. Standalone steppers let the application supply each time step, while
controlled integrators also manage time-step selection. This separation lets
one operator be reused in different algorithms and lets the same algorithm
work with different vector backends.

The sections below are organized from lower-level data representation to
higher-level algorithms. Read the vector and operator sections first when
adding a new FleCSI application; then choose a matrix, solver, or time
integrator based on the mathematical form of the problem. The final physics
section describes optional higher-level helpers that combine these building
blocks for common discretizations.

The source paths in this page are relative to the repository root. The
headers are the most precise reference for template signatures and
defaults; this page focuses on the design and the normal usage
patterns.

Vectors
-------

Vectors provide the common data model consumed by operators, solvers, and
time integrators. A vector has three parts:

``Data``
  Describes where values live and how they are accessed.

``Operations``
  Implements copies, algebraic updates, reductions, and norms for that data.

``Config``
  Supplies scalar and index types and identifies the physics variable.

The common implementation is ``flecsolve::vec::core`` in
``flecsolve/vectors/core.hh``. It exposes operations including
``copy``, ``zero``, ``set_scalar``, ``scale``, ``axpy``, ``linear_sum``,
``dot``, ``l2norm``, ``inf_norm``, and ``global_size``. Algorithms use these
operations instead of assuming a particular storage type.

.. _future-scalars:

Future-valued scalars
~~~~~~~~~~~~~~~~~~~~~

Scalar arguments to ``set_scalar``, ``scale`` (both forms), ``axpy``,
``axpby``, ``linear_sum``, and ``add_scalar`` can be ordinary values,
``flecsolve::future<T>`` values, or
``flecsolve::future_transform<flecsolve::future<T>, F>`` values. The scalar
or underlying future value must be convertible to the vector's ``scalar``
type. ``flecsolve::future<T>`` names a FleCSI single-launch future; it is
not ``std::future``. For operations with two coefficients, either or both
can be futures.

For topology-backed vectors, the operation passes the future and its
scalar transform to the consuming FleCSI task. The task evaluates the
transform when it executes, after the future's value is available. This
allows a reduction result to feed a vector operation without an explicit
``get()`` in application code:

.. code-block:: cpp

   auto alpha = x.dot(x);
   out.axpy(alpha, x, y);       // out = alpha * x + y
   out.scale(x.l2norm());      // the norm's transform runs in the task

``flecsolve/util/future.hh`` defines ``future_transform``, which pairs a
future with a callable. ``flecsolve/util/scalar_ops.hh`` supplies scalar
transforms and multiplication of a FleCSI future by a scalar of the same
type. Bring the multiplication overload into scope outside the
``flecsolve`` namespace:

.. code-block:: cpp

   using flecsolve::operator*;
   auto dt = flecsi::make_future(0.01);
   out.axpy(0.5 * dt, x, y);   // multiply dt by 0.5 in the vector task

These overloads do not provide arbitrary arithmetic on future expressions.
Calling ``get()`` explicitly resolves a future (and applies its transform)
in the caller; reserve that for code that needs the value, such as a host
convergence decision. Custom vector backends must implement the corresponding
future-transform operations to support these scalar arguments.

FleCSI topology views
~~~~~~~~~~~~~~~~~~~~~

``flecsolve::vec::topo_view`` adapts a FleCSI field reference to the vector
interface. Construct one with ``flecsolve::vec::make`` after its topology has
been allocated:

.. code-block:: cpp

   auto solution = flecsolve::vec::make(solution_field(mesh));
   auto work = flecsolve::vec::make(work_field(mesh));

The view retains the topology and field references needed to launch the
distributed vector operations. This is the normal choice for FleCSI
applications. The heat-equation tutorial demonstrates this pattern with two
fields representing the current and next solution.

The view's variable tag can also be used to select a component from a
multi-component vector. A variable is a compile-time label, not a runtime
string; see ``flecsolve/vectors/variable.hh`` and
``flecsolve/vectors/topo_view.hh``.

Other vector backends
~~~~~~~~~~~~~~~~~~~~~

The repository provides several backends:

* ``flecsolve::vec::multi`` groups component vectors into one vector-like
  object. Operators can select the subset of variables they own.
* ``flecsolve::vec::seq`` provides sequential views over contiguous data,
  and is useful inside local sparse-matrix kernels.

Use ``vec::multi`` when a coupled state contains several fields but an
operator acts on only some of them. Use a backend that matches the ownership
and communication model of the underlying data; converting every field to a
new contiguous array usually defeats the purpose of topology-backed vectors.

Operators
---------

An operator maps a domain vector to a range vector. Its essential operation
is:

.. code-block:: cpp

   template<class Domain, class Range>
   void apply(const Domain & x, Range & y) const;

``flecsolve::op::base`` stores optional parameters and declares input and
output variable tags. ``flecsolve::op::core`` adds the vector-facing
``apply``, ``operator()``, and ``residual`` functions. The core wrapper
selects the operator's input and output subsets before forwarding to the
implementation.

A task-backed operator commonly looks like this:

.. code-block:: cpp

   struct diffusion : flecsolve::op::base<parameters> {
     using flecsolve::op::base<parameters>::base;

     template<class Domain, class Range>
     void apply(const Domain & x, Range & y) const {
       flecsi::execute<laplace>(y.data.topo(), params, y.data.ref(),
                                x.data.ref());
     }
   };

   flecsolve::op::core<diffusion> F(coefficients);

The exact task signature depends on the topology and privileges. The
heat-equation operator in ``examples/heat_equation/heat.hh`` is a complete
example of this pattern.

Operator ownership
~~~~~~~~~~~~~~~~~~

Use the handle that matches the lifetime relationship between the algorithm
and the operator:

``op::ref(F)``
  A non-owning mutable reference. Use this when ``F`` is owned by the caller
  and outlives the solver or integrator.

``op::cref(F)``
  A non-owning const reference.

``op::make_shared<Operator>(...)``
  A shared owning handle. This is useful when a solver factory or an
  implicit integrator must retain an operator beyond the local scope.

Handles are defined in ``flecsolve/operators/handle.hh``. Making ownership
explicit avoids accidental copies of large operator state and makes it clear
which objects must remain alive during a solve.

Shell and composed operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``flecsolve/operators/shell.hh`` provides a shell operator for wrapping an
application callback. ``operators/factory.hh`` and
``solvers/factory.hh`` provide runtime selection when an operator or solver
must be chosen from configuration rather than a C++ template parameter.

Matrices
--------

Matrices implement the same operator interface through sparse matrix-vector
multiplication. The base ``flecsolve::mat::sparse`` class delegates its
``apply`` and ``mult`` operations to a backend-specific ``spmv`` operation.
This allows a matrix to be passed anywhere an operator is expected:

.. code-block:: cpp

   matrix.apply(x, y);       // y = A x
   matrix.mult(x, y);        // equivalent sparse matrix-vector product

Sequential sparse matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``flecsolve/matrices/seq.hh`` contains compressed and coordinate formats.
The main types are:

* ``flecsolve::mat::csr`` for compressed sparse row storage;
* ``flecsolve::mat::coo`` for coordinate construction before conversion.

The compressed representation stores offsets, indices, and values. Use COO
when assembling entries is more convenient, then convert to CSR or CSC for
repeated products. The sequential matrix tests in
``flecsolve/matrices/test`` show the supported construction and serialization
helpers.

Distributed CSR
~~~~~~~~~~~~~~~

``flecsolve::mat::parcsr`` in ``flecsolve/matrices/parcsr.hh`` distributes
the rows and columns of a CSR matrix across MPI processes. Its SpMV is split
into local and remote contributions; the remote part is accumulated in a
temporary topology-backed vector. A parallel CSR matrix can be constructed
from a Matrix Market file or from a distributed CSR initialization object.

Use ``parcsr`` when the linear system is naturally represented as a sparse
matrix and the matrix is reused across many products. Use a task-backed
operator instead when assembling a matrix would be expensive or when the
physics operator is better evaluated matrix-free.

Solvers
-------

The solver package treats a linear system as an operator ``A`` and solves

.. math::

   A x = b.

The Krylov solvers share a common setup:

1. Create or read solver settings.
2. Allocate work vectors with the solver's ``make_work`` helper.
3. Construct the solver with the operator and optional preconditioner.
4. Call ``apply(b, x)`` and inspect the returned ``solve_info``.

The available Krylov methods are:

``cg``
  Conjugate gradient for symmetric positive-definite systems.

``gmres``
  Generalized minimum residual for nonsymmetric systems. Its work and
  restart choices should be considered for large problems.

``bicgstab``
  A nonsymmetric method with a smaller work footprint than unrestarted
  GMRES.

``nka``
  Nonlinear Krylov acceleration for fixed-point-like updates.

The solver headers are ``flecsolve/solvers/cg.hh``, ``gmres.hh``,
``bicgstab.hh``, and ``nka.hh``. The runtime registry and factory are in
``flecsolve/solvers/factory.hh``.

Settings and preconditioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common settings include ``maxiter``, ``rtol``, ``atol``, and
``use_zero_guess``. A preconditioner is another operator with an ``apply``
method. The solver applies it to residuals without needing to know how the
preconditioner stores data.

Solver factories and runtime configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the solver type is known at compile time, construct the solver directly
and use the method-specific ``options`` type. When the solver should be
selected from an input file, use ``flecsolve::krylov_factory`` from
``flecsolve/solvers/factory.hh``. The factory registers the supported Krylov
targets and stores the selected solver settings in a type-safe variant.

The factory currently supports ``cg``, ``gmres``, ``bicgstab``, and ``nka``.
Its configuration has two levels: ``type`` selects the solver, and the
``[linear-solver.options]`` section supplies the settings for that solver:

.. code-block:: ini

   [linear-solver]
   type = cg
   [linear-solver.options]
   maxiter = 1000
   rtol = 1e-6
   atol = 1e-8
   use-zero-guess = false

The corresponding C++ setup reads the factory options and passes the vector
and operator needed to construct the selected solver:

.. code-block:: cpp

   auto solver_settings = flecsolve::read_config(
       "solver.cfg", flecsolve::krylov_factory::options("linear-solver"));

   auto solver = flecsolve::krylov_factory::make_shared(
       solver_settings, rhs, A);

The ``rhs`` argument is used to deduce and allocate the solver's work-vector
type. ``A`` is the operator to invert. ``make_shared`` returns a shared
operator handle, so it can be passed to another component, such as an
implicit time integrator, without exposing which concrete Krylov solver was
selected. Use ``krylov_factory::make`` instead when the resulting value can
remain local and does not need shared ownership.

The factory's options object must be read together with the configuration
file that contains its sections. The nested options section is important:
``[linear-solver.options]`` is generated from the selected solver's own
options type. Changing ``type`` from ``cg`` to ``gmres`` therefore changes
which options are validated and how the solver is constructed, while the
call site remains unchanged. The factory example in
``flecsolve/solvers/test/nka-factory.cfg`` demonstrates this pattern in a
larger nonlinear solve.

Factories can also be nested. For example, an application can register
its own diagonal and identity preconditioners with ``op::factory``. This
follows the custom factory in ``flecsolve/solvers/test/nka.cc``:

.. code-block:: cpp

   #include "flecsolve/time-integrators/factory.hh"
   #include "flecsolve/solvers/factory.hh"

   namespace app {
   enum class pc_type { diagonal, identity };

   inline std::istream & operator>>(std::istream & in, pc_type & type) {
     std::string name;
     in >> name;
     if (name == "diagonal") type = pc_type::diagonal;
     else if (name == "identity") type = pc_type::identity;
     else in.setstate(std::ios_base::failbit);
     return in;
   }

   template<pc_type T>
   struct pc_registry {
     using settings = flecsolve::null_settings<T>;
     using options = flecsolve::null_options<T>;

     template<class O>
     static auto make(const settings &, const flecsolve::op::handle<O> & A) {
       if constexpr (T == pc_type::diagonal)
         return A.get().Dinv();
       else
         return flecsolve::op::I.get();
     }
   };

   struct pc_policy {
     using target = pc_type;
     using targets = flecsolve::includes<target::diagonal, target::identity>;
     template<target T> using registry = pc_registry<T>;
   };
   using pc_factory = flecsolve::op::factory<pc_policy>;
   }

The policy supplies the available targets and their registries; each
registry supplies settings, options, and construction. ``null_settings<T>``
and ``null_options<T>`` describe methods that need no additional options.
With that factory, the application can compose BDF, NKA, GMRES, and the
selected preconditioner:

.. code-block:: cpp

   using namespace flecsolve;
   using namespace flecsolve::time_integrator;

   auto [time_s, nonlinear_s, linear_s, pc_s] = read_config(
       "nested.cfg",
       integrator_factory::options("time-integrator"),
       krylov_factory::options("nonlinear-solver"),
       krylov_factory::options("linear-solver"),
       app::pc_factory::options("preconditioner"));

   auto integrator = integrator_factory::make(
       time_s, u, F,
       krylov_factory::make_shared(
           nonlinear_s, u, F,
           krylov_factory::make_shared(
               linear_s, u, J, app::pc_factory::make_shared(pc_s, J))));

Here ``u`` is the prototype solution vector, ``F`` is an application-provided
implicit time operator with residual and linearization support, and ``J``
is a handle to its correction operator providing ``Dinv()``. The application
must keep ``J`` and its diagonal preconditioner consistent with the current
linearization and BDF scaling; factory composition does not refresh them.

The corresponding method-selection excerpt is:

.. code-block:: ini

   [time-integrator]
   type = bdf
   [time-integrator.options]
   integrator = BDF2
   starting-integrator = BE

   [nonlinear-solver]
   type = nka
   [nonlinear-solver.options]
   max-dim = 5

   [linear-solver]
   type = gmres

   [preconditioner]
   type = diagonal

Complete ``nested.cfg`` with the required BDF time/controller settings and
solver tolerances and iteration limits in ``[nonlinear-solver.options]``
and ``[linear-solver.options]``. The C++ wiring fixes the nesting; changing a
factory's ``type`` and matching options selects a compatible alternative
without changing that construction code. For example, switching
``preconditioner.type`` to ``identity`` selects the other custom registry.

For a directly constructed solver, the pattern is:

.. code-block:: cpp

   auto work = flecsolve::op::cg::make_work(u);
   flecsolve::op::cg::parameters parameters(settings, op::ref(A),
                                            op::ref(P), work);
   flecsolve::op::cg::solver solver(parameters);
   auto info = solver.apply(rhs, solution);

The exact parameter constructor can vary by solver. For configuration-driven
applications, prefer ``krylov_factory`` and the common settings/options
interfaces shown in the heat-equation and solver test examples.

Always check ``solve_info``. A returned solution may be useful even when the
solver reaches its iteration limit, but the caller should decide whether to
accept it based on ``status``, ``iters``, and the final residual norm.

Multigrid and AMP integrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``solvers/mg`` directory contains reusable multigrid building blocks:
levels, coarsening, intergrid transfer, smoothers, and cycles. These pieces
are intended for applications that can provide the corresponding hierarchy
and operators.

AMP-backed solvers live in ``solvers/amp.hh`` and ``solvers/amp.cc`` and are
enabled by the project's AMP configuration. They are optional; applications
that do not need them can disable AMP as described in the build guide.

.. _time-integrators:

Time integrators
----------------

Time integrators solve an evolution equation of the form

.. math::

   \frac{d u}{d t} = F(u,t)

by repeatedly calling an operator ``F``. They own their algorithmic work
vectors but operate on the caller's vector type. Each method exposes a
``stepper`` for application-managed stepping and an ``integrator`` with time
bookkeeping and step-size selection. The controlled interface remains
available; the standalone interface does not require a time-step controller.

Standalone Runge--Kutta steppers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rk23::stepper`` and ``rk45::stepper`` expose
``advance(dt, current, next)``. Construct their parameters with an operator
handle and work vectors, without integrator settings:

.. code-block:: cpp

   namespace rk = flecsolve::time_integrator::rk23;
   rk::stepper step(rk::parameters(flecsolve::op::ref(F),
                                   rk::make_work(current)));

   auto dt = flecsi::make_future(0.01);
   step.advance(dt, current, next);
   std::swap(current, next);

Use the ``rk45`` namespace for the higher-order method. Both steppers accept
an ordinary scalar or a compatible FleCSI scalar future for ``dt``. Stage
coefficients derived from a future time step are evaluated inside vector
tasks. There is no need to resolve ``dt`` with ``get()`` before advancing.

The caller chooses each step size, manages time and stopping conditions, and
decides when to accept and swap the solution. These RK steppers have no
``check_solution()``, ``update()``, or ``get_next_dt()`` methods. They still
compute an embedded error vector in the ``workvecs::z`` workspace. Work can
be owned by the parameters or supplied with ``std::ref(work)``; borrowed
work and non-owning operator handles must outlive the stepper.

``time_integrator::stepper_factory`` selects ``rk23``, ``rk45``, or ``bdf``
through ``type`` in the requested configuration section. RK selections need
no method options. Existing RHS-only construction remains available as
``stepper_factory::make(settings, current, rhs)``; selecting BDF through that
call throws ``std::invalid_argument`` because BDF requires a solver.

The factory wrapper also provides ``check_solution()``, ``update()``,
``get_current_step()``, and ``num_step_rejects()``. For RK,
``check_solution()`` always returns true, ``update()`` only increments the
accepted-step count, and the rejection count is zero. Existing RK callers
can still just advance and swap. These methods let applications use the
same acceptance loop for RK and BDF without adding a timestep controller.

Explicit Runge--Kutta
~~~~~~~~~~~~~~~~~~~~~

``flecsolve/time-integrators/rk23.hh`` implements an adaptive explicit
Runge--Kutta method with an error estimate. ``rk45.hh`` provides a higher
order alternative. The controlled ``integrator`` loop is:

.. code-block:: cpp

   integrator.advance(dt, current, next);
   bool accepted = integrator.check_solution();
   if (accepted) {
     integrator.update();
     std::swap(current, next);
   }
   dt = integrator.get_next_dt(accepted);

``advance(dt, current, next)`` writes a candidate solution into ``next``;
it does not accept the step or advance the integrator's clock.
``check_solution()`` checks whether the candidate is acceptable. On acceptance,
call ``update()`` to commit time and integrator state, then swap the vectors
so that the candidate becomes ``current``.

On rejection, obtain a retry timestep with ``get_next_dt(false)``, and
call ``advance`` again.

To select RK23 or RK45 through configuration, include
``flecsolve/time-integrators/factory.hh`` and build the integrator from an
RHS handle with ``integrator_factory``:

.. code-block:: cpp

   using namespace flecsolve;
   using namespace flecsolve::time_integrator;
   auto settings = read_config("explicit.cfg",
       integrator_factory::options("time-integrator"));
   auto integrator = integrator_factory::make(settings, current, op::ref(F));

The ``[time-integrator]`` section sets ``type = rk23`` or ``type = rk45``,
and ``[time-integrator.options]`` holds the method settings. Given only an
RHS handle, selecting ``type = bdf`` throws ``std::invalid_argument``; BDF is
built through the overload that takes a time operator and solver, described
below.

Implicit BDF
~~~~~~~~~~~~

``flecsolve/time-integrators/bdf.hh`` implements variable-step backward
differentiation formulas, including backward Euler, Crank--Nicolson, and
BDF2 through BDF6 modes. Each implicit step requires a linear solve, so the
integrator receives a solver as part of its parameters.

``operator_adapter`` in ``time-integrators/operator_adapter.hh`` wraps a
right-hand-side operator ``F`` and exposes the scaled implicit operator

.. math::

   G(x) = x - \gamma F(x).

The BDF integrator updates ``gamma`` as the time step changes. This lets the
same physics operator be reused while the linear solver sees the system
appropriate for the current step.

To select a controlled implicit integrator through configuration, pass
``integrator_factory`` the time operator and a solver handle:

.. code-block:: cpp

   using namespace flecsolve;
   using namespace flecsolve::time_integrator;
   auto settings = read_config("implicit.cfg",
       integrator_factory::options("time-integrator"));
   auto integrator = integrator_factory::make(settings, current, F, solver);

This overload accepts every method. With ``type = rk23`` or ``type = rk45``,
the RK integrator reads the RHS through ``F.apply_rhs()`` and never calls the
solver, so an application wired for BDF can switch to an explicit method by
configuration alone. Method selection has two levels for BDF: ``type = bdf``
selects the family, and the ``integrator`` option selects BE, CN, or BDF2
through BDF6 within it.

For ``type = bdf``, put BDF settings,
including time bounds, tolerances, ``integrator = BE``, ``CN``, or
``BDF2`` through ``BDF6``, and ``starting-integrator``, in
``[time-integrator.options]``. The factory allocates workspace using
``current`` as a prototype. ``F`` must provide BDF's ``apply_rhs``, scaling,
and validity operations; the application supplies a solver handle that
solves the corresponding implicit system using the same operator state.
Both shared and borrowed operator and solver handles are supported. Borrowed
objects must outlive the integrator.

The factory uses the controlled acceptance loop shown above. BDF's
``advance(dt, current, next)`` takes the startup path while no step has been
accepted. Rejected attempts do not advance the accepted-step count, so
startup continues until ``update()`` commits the first accepted step.
``current`` and ``next`` must be distinct vectors.

Standalone BDF stepper
~~~~~~~~~~~~~~~~~~~~~~

``bdf::stepper`` separates the implicit step and solution history from the
controlled integrator's clock and next-step selection. Read method settings
with ``bdf::stepper_options`` and construct it with the RHS operator handle,
work vectors, and a solver handle configured for the implicit operator:

.. code-block:: cpp

   namespace bdf = flecsolve::time_integrator::bdf;
   auto settings = flecsolve::read_config(
       "stepper.cfg", bdf::stepper_options("time-integrator"));
   bdf::stepper step(bdf::parameters(settings, F,
                                     bdf::make_work(current), solver));

   step.advance(dt, current, next);
   if (step.check_solution()) {
     step.update();
     std::swap(current, next);
   }

Here ``F`` and ``solver`` are operator handles. Unlike the RK steppers,
BDF's ``advance`` takes only a ``double`` time step, and ``current`` and
``next`` must be distinct vectors. Startup lasts until the first accepted
step. Call ``update()`` only
for accepted steps to commit the multistep history. After rejection, retain
``current`` and choose a retry step size in the application.

The BDF stepper retains solver-success and optional truncation-error checks,
predictor settings, and history; it does not expose ``get_next_dt()`` or
maintain the application's current/final time. ``stepper_options`` omits
the base time-range and step-limit options, but still includes BDF method,
predictor, and error-estimation settings. Use ``bdf::integrator`` and
``bdf::options`` when built-in time bookkeeping and step selection are
wanted. See ``flecsolve/time-integrators/test/implicit.cc`` and its
``implicit.cfg`` for a complete setup.

BDF can also be constructed through the shared stepper factory:

.. code-block:: cpp

   using namespace flecsolve;
   using namespace flecsolve::time_integrator;
   auto settings = read_config("stepper.cfg", stepper_factory::options("stepper"));
   auto step = stepper_factory::make(settings, current, F, solver);

   step.advance(dt, current, next);
   if (step.check_solution()) {
     step.update();
     std::swap(current, next);
     time += dt; // The application owns the clock.
   }
   // On rejection, keep current and time unchanged and choose a retry dt.

Select ``type = bdf`` in ``[stepper]``, and put the existing
``bdf::stepper_options`` in ``[stepper.options]``. For example:

.. code-block:: ini

   [stepper]
   type = bdf
   [stepper.options]
   integrator = BE
   starting-integrator = BE
   timestep-selection-strategy = constant
   predictor = leapfrog
   use-predictor = false
   calculate-time-trunc-error = false

There are no required time bounds, initial timestep, or step limit for a
standalone stepper. ``F`` and ``solver`` share the implicit operator state,
as with ``integrator_factory``. The solver-taking factory overload supports
all three targets: RK selections use ``F.apply_rhs()`` and leave the solver
unused, while BDF uses the implicit system. This permits changing between
RK and BDF through configuration with the same application wiring.

The BDF factory wrapper derives startup from its accepted-step count, so a
rejected first attempt remains in startup. Call ``update()`` only after
acceptance to commit solution history and the previous timestep used by
predictors. BDF accepts scalar timesteps; passing a future when BDF is
selected throws ``std::invalid_argument``. RK retains scalar and future
support through both construction overloads. The factory provides no
``get_next_dt()`` or clock: the application chooses retry and subsequent
timesteps and stopping conditions. See
``flecsolve/time-integrators/test/stepper_factory.cc`` and its
``stepper-factory.cfg`` for variable-step BDF comparisons and rejected-step
retries.

Configuration
~~~~~~~~~~~~~

``flecsolve::util::read_config`` reads Boost program-options settings from a
configuration file. Integrator options are grouped in a named section such
as ``[time-integrator]``; implicit solves commonly add a ``[linear-solver]``
section. The example files are:

* ``examples/heat_equation/explicit.cfg``
* ``examples/heat_equation/implicit.cfg``
* ``flecsolve/time-integrators/test/explicit.cfg``
* ``flecsolve/time-integrators/test/implicit.cfg``

Physics helpers
---------------

The ``flecsolve/physics`` tree builds higher-level pieces from the same
vector and operator abstractions:

``physics/boundary``
  Dirichlet, Neumann, and Robin boundary-condition interfaces.

``physics/volume_diffusion``
  Coefficient and diffusion operators for volume-based discretizations.

``physics/reaction``
  Reaction-rate and mechanism helpers, including Arrhenius and FKN models.

``physics/specializations``
  FleCSI topology specializations such as finite-volume ``narray`` support.

These components are useful when the application's discretization matches
their assumptions. Otherwise, implement a small application-specific
operator and retain the generic solver and time-integrator layers.
