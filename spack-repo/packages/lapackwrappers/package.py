from spack.package import *


class Lapackwrappers(CMakePackage):
    homepage = "https://asc-git.lanl.gov/xcap/oss/solvers/lapackwrappers"
    git = "ssh://git@re-git.lanl.gov:10022/xcap/oss/solvers/lapackwrappers.git"

    version("main", branch="main")

    variant("shared", default=False, description="shared libraries")

    depends_on("blas")
    depends_on("lapack")

    depends_on("netlib-lapack~shared", when="~shared ^[virtuals=blas,lapack] netlib-lapack")
    depends_on("netlib-lapack+shared", when="+shared ^[virtuals=blas,lapack] netlib-lapack")

    def cmake_args(self):
        args = [
            self.define("INSTALL_DIR", self.prefix),
            self.define("BUILD_TESTING", self.run_tests),
            self.define_from_variant("ENABLE_SHARED", "shared"),
        ]
        if "^intel-mkl" in self.spec:
            args.append(self.define("LAPACK_INSTALL_DIR", self.spec["lapack"].prefix.mkl))
        elif "^intel-oneapi-mkl" in self.spec:
            args.append(self.define("LAPACK_INSTALL_DIR", self.spec["intel-oneapi-mkl"].package.component_prefix))
        else:
            args.append(self.define("LAPACK_INSTALL_DIR", self.spec["lapack"].prefix))

        blas, lapack = self.spec["blas"].libs, self.spec["lapack"].libs
        args.extend(
            [
                self.define("BLAS_LIBRARY_NAMES", ";".join(blas.names)),
                self.define("BLAS_LIBRARY_DIRS", ";".join(blas.directories)),
                self.define("LAPACK_LIBRARY_NAMES", ";".join(lapack.names)),
                self.define("LAPACK_LIBRARY_DIRS", ";".join(lapack.directories)),
            ]
        )
        return args

