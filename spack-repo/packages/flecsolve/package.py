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

    depends_on('flecsi@2.2:')
    depends_on('amp')
    depends_on('stacktrace+shared')
    depends_on('lapackwrappers@main')

    def cmake_args(self):
        args = [
            self.define_from_variant("FLECSOLVE_ENABLE_UNIT_TESTS", "tests"),
        ]
        return args

