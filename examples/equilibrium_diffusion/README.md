Example for `multivector` equilibrium diffusion

**Overview**
This example shows how to construct a `multivector` using `enum` types that refer to seperate subvectors, and how operators may be built that are `apply()`'d to a specific subvector or the full `multivector`.

**Mesh**
In this example, we set up 2D mesh using input options to define the mesh extents. The underlying data structure for the mesh is a specialization of the `flecsi::narray` topology.

**Problem variables**
The variables of the problem are defined with

    enum class diffusion_var { v1 = 0, v2 };
where `v1`,`v2` refer to variables 1 and 2 of the multivector.

**Vectors**
We will use flecsi fields to hold the `vector` data, as in

    // definitions for a flecsi field to store vector data
    template<auto S>
    using fld = const field<scalar_t>::definition<msh, S>;
    // we will construct our multivector from a collection of fields.
    std::array<fld<msh::cells>, NVAR> xd{}

The driver routine constructs the multivectors from the array of fields

    auto X = make_multivector(xd);
    auto RHS = make_multivector(rhsd);

where `make_multivector` is a utility function defined in the example for convenience. This function associates the `diffusion_var::v1` with the `0` index of the field array, `diffusion_var::v2` with `1` index, and so on.

We can retrieve references the `vectors` themselves using

    auto & [vec1, vec2] = X;

We can manipulate the vectors as `multivectors` or as `vectors`,

    // set both to scalar
    X.set_scalar(2.0);
    // set each to explicitly
    vec1.set_scalar(1.0);
    vec2.set_scalar(0.0);

**Operators**

***Important Note:** In our implementation, we have designed the operators to be composable as operator expressions. This is **not** a requirement to use the solvers library, and our particular operators implementations do not impose requirements downstream.
The only requirements of operators is that they posses the operator interface: `apply()`, `reset()`, and `get_parameters()`. We demonstrate here only one possible way to implement them.
**Our implementation still under construction**, and subject to further refinement and alteration in the future.*

We will define operators for
 - Boundary conditions
 - Volume Diffusion

Each operator is parameterized on one or more `diffusion_var` variables that can be implied through the `vector` or explicitly provided.

    auto bnd_op_1 =
    		flecsolve::physics::op_expr(flecsolve::multivariable<diffusion_var::v1>,
    	                                make_boundary_operator_dirichlet(vec1),
    	                                make_boundary_operator_pseudo(vec1)

This makes a boundary-value operator for `diffusion_var::v1` (with associated vector `vec1`).

Using expressions to make other expressions, we construct the full `multivector` operator as

    auto A = flecsolve::physics::op_expr(
    		flecsolve::multivariable<diffusion_var::v1, diffusion_var::v2>,
    		bnd_op_1,
    		make_volume_operator<0>(vec1, diff_param_beta[0], diff_param_alpha[0]),
    		bnd_op_2,
    		make_volume_operator<1>(vec2, diff_param_beta[1], diff_param_alpha[1]));

To parse this a little, first we pass in a set of variables we want this operator to operate on. As this is the top-level operator we use in the solver, we give it all the variables of `diffusion_var`.
Next, we give it operators for the singular vectors of those variables. `bnd_op_1` is the previously defined operator expression, but we also construct a new operator for the volume diffusion with `make_volume_operator<>()`.
In this example, we construct a volume diffusion operator as a `coefficent` setter (which sets the diffusion coefficients on cell faces) with a `diffusion` operator).

**Solver**
To solve the system, we first construct the solver parameters. In this example, we will use a CG solver as

    flecsolve::op::krylov_parameters params(
    		flecsolve::cg::settings("solver"),
    		flecsolve::cg::topo_work<>::get(RHS),
    		std::ref(A));
To fill the parameters object, we load a configuration

    read_config("diffusion.cfg", params);
The file `diffusion.cfg` defines the tolerances and limits used in the CG solve.
Finally, we create the solver as

    auto info = slv.apply(RHS, X);
This will solve `AX=RHS`, and store the result in `X`.

We can query some details of the solve

    flog(info) << "norm = " << info.res_norm_final << "\n";
    flog(info) << "iters = " << info.iters << "\n";

The results of the `apply()` are finally outputted to text output, of the format
`var: i, j, k, x, y, z, value[k][j][i]`
for each processor rank.

