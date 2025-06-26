from spack.package import *
from spack.pkg.builtin.flecsi import Flecsi

class Flecsi(Flecsi):
    version("2.4.0", tag="v2.4.0", commit="598d518b4105ec91ee42ee50420aa46a32a0f60f")
