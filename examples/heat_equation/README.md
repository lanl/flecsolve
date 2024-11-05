# Heat Equation in 2D

This example considers,
```math
\begin{align*}
\frac{\partial u}{\partial t} &= \alpha \Delta u  & \text{in} & \ \Omega = (0,10) \times (0,10) \ \text{for} \ t > 0\\
u &= 0                                            & \text{on} & \ \partial \Omega \ \text{for} \ t > 0\\
u &= f(x,y)                                       & \text{in} & \ \Omega \ \text{at} \ t = 0,
\end{align*}
```
where $`f`$ is given by
```math
f(x,y) =
\begin{cases}
50 &\text{if } 4 \le x \le 6, 4 \le y \le 6,\\
0 &\text{else.}
\end{cases}
```

**Running** The binaries take the mesh extents as positional
paramters. Whether to output the solution at every step and $`\alpha`$
are additional optional parameters.

For example, to run with implicit time integration using $`\alpha = 1.5`$ and outputting the solution at each step:
```bash
$ mpirun -np 4 ./heat-implicit 100 100 -d 1.5 -o true
```
Similarly, to run with explicit time integration:
```bash
$ mpirun -np 4 ./heat-explicit 100 100 -d 1.5 -o true
```

The utilities in [utility directory](./util) can be used to visualize the solutions.
![solution](./figs/solution.gif "Solution")

**Mesh** The mesh is a simple 2D specialization of the `narray`
topology [mesh](./mesh.hh).  In addition to defining the mesh,
variables are declared for the mesh and coloring slots, as well as
field definitions that will be used to store $`u^k`$ and $`u^{k+1}`$.

    inline mesh::slot m;
    inline mesh::cslot coloring;

    inline std::array<field<double>::definition<mesh, mesh::vertices>, 2> ud;

**Control Policy** The control policy for this example can be found in
[control policy](./control.hh). `initialize`,`advance`, and `finalize`
control points are declared and state is added to store the
`flecsolve::vec::topo_view` instances.  These provide a vector interface to
the fields stored on mesh and will be backed by the field definitions
declared previously.  A helper function is added that will be used to
intialize these once the mesh has been allocated:

    void initialize_vectors() {
	    u_.emplace(flecsolve::vec::make(m, ud[0](m)));
	    unew_.emplace(flecsolve::vec::make(m, ud[1](m)));
    }

**Operator** The operator used to apply $`\alpha \Delta`$ to a vector is defined in [heat.hh](./heat.hh):

    struct heat_op : flecsolve::op::base<heat_params> {
        heat_op(double d) : flecsolve::op::base<heat_params>(d)) {}

        template<class Domain, class Range>
        void apply(const Domain & x, Range & y) const {
            flecsi::execute<task::laplace>(y.data.topo(),
                                           diffusivity,
                                           y.data.ref(),
                                           x.data.ref());
        }
    };

The operator needs to provide what variables it operates on and
provide an `apply`.  In this case, the operator uses anonymous input
and output variables and the apply simply calls the `task::laplace`
task using the topology slot and field references requested from the
given vectors.

**Driver** The driver that contains the common control point actions
and `main` function for the explicit and implicit binaries is
[heat.cc](./heat.cc).  The `init_mesh` control point allocates the
mesh and then uses the control policy object `cp` to save information
about the mesh and initialize the vectors.

    m.allocate(mesh::mpi_coloring(idef), geometry);

    cp.diffusivity = diffusivity.value();
    cp.initialize_vectors();
    cp.save_geometry(geometry, axis_extents);

**Time Integration** The explicit time integration can be found in
[explicit.cc](./explicit.cc).  First the time integrator is created from
its parameters:

    rk23::integrator ti(rk23::parameters(
	    read_config("explicit.cfg", rk23::options("time-integrator")),
		op::ref(F),
		rk23::make_work(u)));

The first parameter to `rk23::parameters` is the runtime settings for the integrator.
These are given by reading a configuration file.  In this example, `explicit.cfg` is the
filename of the configuration file and `"time-integrator"` is the named section of the
configuration file specifying the options for the integrator.  The second parameter to
`rk23::parameters` is a handle to the operator used for the time integrator.  The last
parameter is an array of work vectors needed by `rk23`.  `rk23::make_work` is a helper
for automatically creating this workspace array given a template vector.

Finally, the time integration loop uses $`u^k`$ and $`u^{k+1}`$ (called `u` and
`unew`) to integrate the solution in time:

    auto dt = ti.get_current_dt();
    while (ti.get_current_time() < ti.get_final_time()) {
        ti.advance(dt, u, unew);
        auto good_solution = ti.check_solution();
        if (good_solution) {
            ti.update();
            std::swap(u, unew);
        }
        dt = ti.get_next_dt(good_solution);
    }


---
The implicit time integration is implemented in
[implicit.cc](./implicit.cc).  The majority of this resembles the
explicit time integration.  One main difference is the parameters
given to the `bdf` integrator:

    auto F = op::make_shared<operator_adapter<heat_op>>(cp.diffusivity);
    bdf::integrator ti(
		bdf::parameters(ti_settings,
		                F,
		                bdf::make_work(u),
	                    krylov_factory::make_shared(slv_settings, u, F)));

Instead of giving the implicit integrator the `heat_op` operator
directly, an `operator_adapter` is used to provide the operator that
will be inverted during the time integration.  Specifically, the
adapter takes the heat operator $`F = \alpha \Delta`$ and computes:
$`(I - \gamma F)`$ where $`\gamma`$ is parameterized scaling from the
time integrator (e.g., $`\Delta t`$ for backward Euler).  The
`bdf::parameters` also takes a solver as its final parameter.  In this
case, a factory is used to construct the specified solver based on the
options specified in the input file.
