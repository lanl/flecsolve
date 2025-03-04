from spack.package import *

class Flecsolve(CMakePackage):
    """Solvers package built on top of FleCSI"""

    homepage="https://re-git.lanl.gov/xcap/oss/flecsolve/"
    git = "ssh://git@re-git.lanl.gov:10022/xcap/oss/flecsolve.git"

    version("main", branch="main")

    variant("tests", default=False, description="Enable unit tests")

    depends_on('flecsi@2.2:')
    depends_on('amp')
    depends_on('stacktrace+shared')

    def cmake_args(self):
        args = [
            self.define_from_variant("FLECSOLVE_ENABLE_UNIT_TESTS", "tests"),
        ]
        return args

