#pragma once

#include <array>

#include <flecsi/flog.hh>
#include <iostream>

#include "flecsolve/vectors/mesh.hh"
#include "flecsolve/vectors/multi.hh"

#include "flecsolve/physics/boundary/dirichlet.hh"
#include "flecsolve/physics/boundary/neumann.hh"
#include "flecsolve/physics/expressions/operator_expression.hh"
#include "flecsolve/physics/volume_diffusion/volume_diffusion.hh"
#include "flecsolve/solvers/cg.hh"
#include "flecsolve/solvers/solver_settings.hh"
#include "flecsolve/physics/specializations/operator_mesh.hh"

#include "test_setup.hh"

using namespace flecsi;

namespace flecsolve {
namespace physics_testing {

constexpr std::size_t NX = 8;
constexpr std::size_t NY = 8;

enum class diffusion_var { v1, v2 };

constexpr scalar_t DEFAULT_VAL = 1.0;

constexpr auto diff_alpha = scalar_t{0.0};
constexpr auto diff_beta = DEFAULT_VAL;

const field<scalar_t>::definition<msh, msh::cells> v1d, v2d, rhs1d, rhs2d,
	diffa;
util::key_array<const field<scalar_t>::definition<msh, msh::faces>, msh::axes>
	diffb;

}
}