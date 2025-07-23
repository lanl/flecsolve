# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *

class Flecsolve(CMakePackage):
    """Solvers package built on top of FleCSI"""

    homepage="https://github.com/lanl/flecsolve.git"
    git = "https://github.com/lanl/flecsolve.git"

    version("main", branch="main")

    variant("tests", default=False, description="Enable unit tests")

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("fortran", type="build")

    depends_on('flecsi@2.4:')
    depends_on('amp+hypre')
    depends_on('stacktrace+shared')
    depends_on('eigen')

    def cmake_args(self):
        args = [
            self.define_from_variant("FLECSOLVE_ENABLE_UNIT_TESTS", "tests"),
        ]
        return args

