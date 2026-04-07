from kessel.workflows import *
from kessel.workflows.base.cmake import CTest as CTestWorkflow
from kessel.workflows.cmake import Build
from pathlib import Path


class Cdash(Build, CTestWorkflow):
    steps = ["env", "configure", "build", "test", "install", "submit"]
    ctest_project_name = environment("flecsolve", variable="CTEST_PROJECT_NAME")
