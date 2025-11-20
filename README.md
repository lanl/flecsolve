The flecsolve package is a parallel computational framework for
multi-physics application development using the open source [FleCSI](https://flecsi.github.io/flecsi/)
programming system.  Flecsolve employs design principles from the [AMP](https://github.com/AdvancedMultiPhysics/AMP)
package to provide a lightweight set of tools for composing
multi-physics applications in FleCSI.  Specifically, flecsolve
provides a linear algebra interface to the FleCSI data abstraction.
This interface is used to implement a set of time integrators and
solvers for FleCSI applications.

# Library Structure
Components in flecsolve are organized into the following categories:

[vectors](flecsolve/vectors)
: Definition of core vector type and specialization of this type for vectors using FleCSI data and vectors with multiple components.

[operators](flecsolve/operators)
: Definition of core operator type used as interface for solvers and time integrators.

[matrices](flecsolve/matrices)
: Implementation of various sequential sparse matrix types and a parallel Compressed Sparse Row (CSR) matrix.

[solvers](flecsolve/solvers)
: Collection of sparse linear solvers.

[time-integrators](flecsolve/time-integrators)
: Collection of variable step explicit and implicit integrators.

# Linear Algebra Interface
The [core](flecsolve/vectors/core.hh) vector type provides a vector interface using composition of three types:

- **Data**: interface to the storage
- **Operations**: implementation of common vector operations
- **Variable**: static tag to identify a physics field

[`vec::topo_view`](flecsolve/vectors/topo_view.hh) specializes this type to adapt user
fields defined on a FleCSI topology to the vector interface.  The data
interface for this vector implementation stores stores a field
reference and provides interfaces to the topology instance, accessor
types, and index space for the field.  The operations type executes
FleCSI tasks to implement common vector operations using the vector
data.

[`vec::multi`](flecsolve/vectors/multi.hh) implements the vector interface for a
collection of component vectors.  This enables a natural grouping of
physics components and permits operators to select a subset of the
multivector on which to operate.

Operators in flecsolve fundamentally define a map from a domain vector
to a range vector.  This map is given by defining a type with an
`apply` function and implementing the desired map through FleCSI task
execution or the provided vector operations.  Optionally, input and
output variables may be specified to indicate a subset of the physics
on which to operate.  A flecsolve operator is then constructed through
specialization of [`op::core`](flecsolve/operators/core.hh) with this
type.

Krylov solvers and time integrators provided by flecsolve are
implemented generically using the vector operations and operator
interface.  This enables them to be used with a variety of FleCSI
topologies, automatically deducing the user types when field
references are wrapped in
[`vec::topo_view`](flecsolve/vectors/topo_view.hh).

# Krylov Solvers
Krylov solvers in flecsolve are defined using a collection of types
with common names in their respective namespace.  These types are then
used to construct a flecsolve operator that computes the approximate
inverse mapping.  These types are summarized below.

`settings`
: Configuration settings for the solver (e.g., tolerances, maximum iteration counts, ...).

`options`
: Specification of boost `options_description` for Krylov solver and
  mapping to its `settings` type.  This is used with `read_config` to
  populate the `settings` type dynamically from a config file.

`make_work`
: Given an exemplar vector used for type deduction, creates the
  appropriate number of work vectors needed during a solve.

`solver`
: Entry point for each Krylov solver containing type information in
  addition to storing the solver settings and auxiliary work vectors
  needed during a solve.  The appropriate number of work vectors needs
  to be be provided as FleCSI fields cannot be constructed dynamically
  during a solve.  To solve a given problem, the `solver` type
  includes a function call operator that returns an approximate
  inverse in the form of a flecsolve operator.  This function takes
  [handles](#operator-handles) to operator to invert and optionally a
  preconditioner and [diagnostic function](#diagnostic-functions) as
  input.

## Operator Handles
[Operator handles](flecsolve/operators/handle.hh) are used to make
ownership explicit when passing flecsolve operators to a solver or
time integrator.  For shared ownership semantics, a flecsolve operator
is constructed using `op::make_shared`.  This returns a handle with
shared ownership.  For a non-owning reference to an operator,
`op::ref` or `op::cref` are used to construct an operator handle.

## Diagnostic Functions
Users may optionally provide a diagnostic function to a Krylov solver
when binding operators to the solver. This allows users to monitor
solver convergence and provide an early termination condition.  A
diagnostic function can be a callable of the following form:

```
bool diagnostic(const Vector & current_solution, double residual_norm)
```

The return value is used to indicate early termination based on the inputs.

# Release

This software has been approved for open source release and has
been assigned **O4869**.

# License

This program is Open-Source under the [BSD-3 License](./LICENSE).
