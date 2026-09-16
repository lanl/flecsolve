flecsolve Documentation
=======================

flecsolve is a parallel computational framework for multi-physics
application development using the open source `FleCSI
<https://flecsi.github.io/flecsi/>`_ programming system. It provides a
linear algebra interface to FleCSI data abstractions and uses that
interface to implement reusable vectors, operators, matrices, time
integrators, and solvers for FleCSI applications.

The project follows design principles from the `AMP
<https://github.com/AdvancedMultiPhysics/AMP>`_ package while keeping the
core solver interfaces lightweight and composable for FleCSI-based
applications.

Library Structure
-----------------

**Vectors**
  Core vector interfaces and implementations for FleCSI topology-backed
  fields, multi-component vectors, and PETSc-backed vectors.

**Operators**
  Interfaces for maps between vector spaces, including ownership-aware
  operator handles used by solvers and time integrators.

**Matrices**
  Sequential sparse matrix types and a parallel compressed sparse row
  matrix implementation.

**Solvers**
  Krylov solvers, nonlinear acceleration utilities, and multigrid
  components.

**Time Integrators**
  Variable-step explicit and implicit integrators that operate on the
  flecsolve vector and operator abstractions.

Release
-------

This software has been approved for open source release and has been
assigned **O4869**.

License
-------

flecsolve is open source under the BSD-3-Clause license. See the
repository ``LICENSE`` file for the complete terms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   build
   user-guide
   components
   examples
   heat-equation-tutorial
   release-notes
