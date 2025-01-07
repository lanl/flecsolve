from spack import *

class Flecsolve(CMakePackage):
    """Solvers package built on top of FleCSI"""
    
    homepage="https://re-git.lanl.gov/xcap/ec/flecsolve/"
    git = "ssh://git@re-git.lanl.gov:10022/xcap/ec/flecsolve.git"

    version("main", branch="main")

    variant("tests", default=False, description="Enable unit tests")

    depends_on('flecsi@2.2:')
    depends_on('amp')

    def cmake_args(self):
        args = [
            self.define_from_variant("ENABLE_UNIT_TESTS", "tests"),
        ]
        return args

