from spack import *

class Flecsolve(CMakePackage):
    """Solvers package built on top of FleCSI"""
    
    homepage="https://re-git.lanl.gov/xcap/ec/flecsolve/"
    git = "ssh://git@re-git.lanl.gov:10022/xcap/ec/flecsolve.git"

    version("develop", branch="main")

    depends_on('flecsi@2.2:')

