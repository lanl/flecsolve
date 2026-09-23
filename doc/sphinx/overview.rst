Overview
========

flecsolve provides reusable numerical building blocks for applications
that already use FleCSI for data, execution, and control flow. The main
interfaces are intentionally generic: solvers and time integrators work
with any vector type that provides the expected vector operations, and
operators define mappings through an ``apply`` interface.

Vectors
-------

The core vector type is built from three pieces:

**Data**
  Storage and access to the underlying values.

**Operations**
  Common linear algebra operations such as copy, scaling, vector sums,
  dot products, and norms.

**Variable**
  A static tag that identifies the physics field represented by the
  vector.

The ``flecsolve::vec::topo_view`` adapter maps fields on a FleCSI
topology to the vector interface. The ``flecsolve::vec::multi`` type
groups component vectors so physics packages can pass coupled state
through one vector-like object while allowing operators to select the
subset they own.

Operators
---------

Operators define maps from a domain vector to a range vector. A user
operator implements the desired map in an ``apply`` function, typically
by launching FleCSI tasks or by composing existing vector operations.

Operator handles make ownership explicit:

**Shared handles**
  Use ``flecsolve::op::make_shared`` to create a shared owning handle.

**Mutable references**
  Use ``flecsolve::op::ref`` to create a non-owning mutable reference
  handle.

**Const references**
  Use ``flecsolve::op::cref`` to create a non-owning const reference
  handle.

Solvers
-------

The solver components include conjugate gradient, BiCGSTAB, GMRES,
nonlinear Krylov acceleration, AMP-backed solvers, and multigrid pieces.
Krylov solver namespaces follow a common structure:

**Settings**
  Runtime configuration such as tolerances and iteration limits.

**Options**
  Boost program-options definitions used to populate settings from a
  configuration file.

**Work-vector factories**
  Factories used to allocate all vectors required during a solve before
  the solve begins.

**Solver object**
  The entry point that binds an operator, optional preconditioner, and
  optional diagnostic callback into an approximate inverse operator.

The corresponding identifiers are generally named ``settings``,
``options``, ``make_work``, and ``solver`` in each solver namespace.

Time Integrators
----------------

The time integration package includes adaptive explicit Runge-Kutta
methods and implicit BDF support. Like the solvers, these integrators are
implemented against the vector and operator interfaces so they can be
used with application-specific FleCSI topology and field types.
