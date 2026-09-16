Build and Install
=================

flecsolve is a CMake project that requires a C++17 compiler, FleCSI, and
Eigen. AMP support is enabled by default and requires AMP and its TPLs.

Dependencies
------------

Required dependencies:

* CMake 3.23 or newer
* C++17 compiler
* FleCSI
* Eigen3

Optional dependencies:

* AMP, enabled with ``FLECSOLVE_ENABLE_AMP``
* TPLs required by AMP

Configure
---------

Configure a release build with AMP enabled:

.. code-block:: console

   cmake -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=/path/to/install

Disable AMP support when the AMP-backed wrappers are not needed:

.. code-block:: console

   cmake -S . -B build \
     -DFLECSOLVE_ENABLE_AMP=OFF

Build and Test
--------------

Build the library:

.. code-block:: console

   cmake --build build

Enable and run unit tests:

.. code-block:: console

   cmake -S . -B build -DFLECSOLVE_ENABLE_UNIT_TESTS=ON
   cmake --build build
   ctest --test-dir build

Build examples:

.. code-block:: console

   cmake -S . -B build -DFLECSOLVE_BUILD_EXAMPLES=ON
   cmake --build build

Install
-------

Install the library, headers, and CMake package files:

.. code-block:: console

   cmake --install build

Downstream CMake projects can then use:

.. code-block:: cmake

   find_package(flecsolve REQUIRED)
   target_link_libraries(my_target PRIVATE flecsolve::flecsolve)

Spack
-----

The repository includes Spack package definitions under ``spack-repo``.
The v2 package currently tracks the ``main`` branch and depends on
FleCSI 2.4 or newer, AMP with Hypre and shared libraries, Stacktrace,
and Eigen.
