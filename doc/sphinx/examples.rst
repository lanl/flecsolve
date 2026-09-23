Examples
========

The repository includes several examples that demonstrate how flecsolve
components are used inside FleCSI applications.

Heat Equation
-------------

``examples/heat_equation`` contains explicit and implicit heat-equation
drivers, configuration files, and utilities for generating VTK output
and animations.

For a guided walkthrough of the problem setup, mesh-backed vectors,
discrete operator, and time integration, see the :doc:`heat-equation-tutorial`.

Important files:

* ``examples/heat_equation/explicit.cc``
* ``examples/heat_equation/implicit.cc``
* ``examples/heat_equation/heat.hh``
* ``examples/heat_equation/explicit.cfg``
* ``examples/heat_equation/implicit.cfg``

Poisson
-------

``examples/poisson`` contains a Poisson example with mesh, control, and
configuration support.

Equilibrium Diffusion
---------------------

``examples/equilibrium_diffusion`` provides a small diffusion example
with source under ``src`` and public headers under ``include``.

Building Examples
-----------------

Enable examples at configure time:

.. code-block:: console

   cmake -S . -B build -DFLECSOLVE_BUILD_EXAMPLES=ON
   cmake --build build
