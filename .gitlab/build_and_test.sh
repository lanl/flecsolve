#!/bin/bash
###############################################################################
export PROJECT_NAME=flecsolve
export PROJECT_DEFAULT_BRANCH=main
export PROJECT_TYPE=oss
export PROJECT_GROUP=oss
###############################################################################

export BUILD_DIR=${BUILD_DIR:-build}
export SOURCE_DIR=${CI_PROJECT_DIR:-$PWD}
TMPDIR=${TMPDIR:-/tmp/xcap/${PROJECT_GROUP}/${PROJECT_NAME}}
CI_SPACK_ENV=${CI_SPACK_ENV:-$TMPDIR/$USER-ci-env}
export CTEST_MODE=${CTEST_MODE:-Continuous}
SUBMIT_TO_CDASH=${SUBMIT_TO_CDASH:-false}
BUILD_WITH_CTEST=${BUILD_WITH_CTEST:-${SUBMIT_TO_CDASH}}
SUBMIT_ON_ERROR=${SUBMIT_ON_ERROR:-${SUBMIT_TO_CDASH}}
SHOW_HELP_MESSAGE=${SHOW_HELP_MESSAGE:-true}

if ${SUBMIT_TO_CDASH}; then
  UNTIL=${UNTIL:-submit}
else
  UNTIL=${UNTIL:-install}
fi

if ${SUBMIT_ON_ERROR}; then
  REPORT_ERRORS="ReportErrors"
else
  REPORT_ERRORS=""
fi
PROJECT_SPACK_ENV_VERSION=${PROJECT_SPACK_ENV_VERSION:-current}

# colors
COLOR_BLUE='\033[1;34m'
COLOR_MAGENTA='\033[1;35m'
COLOR_CYAN='\033[1;36m'
COLOR_PLAIN='\033[0m'

section() {
  if [ -z "${CI}" ]; then
    echo -e "${3:+${COLOR_CYAN}$3${COLOR_PLAIN}}"
  else
    echo -e $'\e[0K'"section_$1:$(date +%s):$2"$'\r\e[0K'"${3:+${COLOR_CYAN}$3${COLOR_PLAIN}}"
  fi
}

print_usage() {
  echo "usage: build_and_test [-h] [-u {spack,env,cmake,build,test,install}] system environment"
  echo
  echo "After the 'env' phase has run, you will be in an active Spack environment."
  echo "You can then use the 'activate_build_env' command to enter a sub-shell"
  echo "with the Spack build environment for your project."
  echo
  echo "To enable CDash submission, set environment variable SUBMIT_TO_CDASH=true"
  echo
  echo "Use the environment variable PROJECT_SPACK_ENV to select a different"
  echo "upstream. You can also configure the AMP Data location by setting"
  echo "the AMP_DATA environment variable."
  echo
  echo "positional arguments:"
  echo "  system         name of the system"
  echo "  environment    name of the Spack environment"
  echo
  echo "options:"
  echo "  -u {spack,env,cmake,build,test,install,submit}, --until {spack,env,cmake,build,test,install,submit}"
  echo "                 run script until the given phase"
  echo "  -h, --help     show this help message and exit"
}

VALID_CMD=false
if [ $# -eq 2 ] && [[ ! "$1" =~ ^-.*  ]] && [[ ! "$2" =~ ^-.*  ]]; then
  VALID_CMD=true
elif [ $# -eq 4 ]; then
  if [ "$1" = "-u" ] || [ "$1" = "--until" ]; then
    if [[ "$2" =~ ^(spack|env|cmake|build|test|install)$ ]]; then
      UNTIL=$2
      VALID_CMD=true
      shift 2
    fi
  fi
fi

if ! $VALID_CMD; then
    print_usage
    false
    return
fi

export SYSTEM_NAME=$1
export SPACK_ENV_NAME=$2

if [[ "$SYSTEM_NAME" == "darwin" || "$SYSTEM_NAME" == "rocinante" || "$SYSTEM_NAME" == "venado" ]]; then
  PROJECT_SPACK_ENV=${PROJECT_SPACK_ENV:-/usr/projects/xcap/spack-env/${PROJECT_TYPE}/${PROJECT_SPACK_ENV_VERSION}}
  AMP_DATA=${AMP_DATA:-/usr/projects/xcap/spack-env/AMP-Data.tar.gz}
elif [[ "$SYSTEM_NAME" == "rzadams" || "${SYSTEM_NAME}" == "rzansel" || "$SYSTEM_NAME" == "rzvernal" ]]; then
  PROJECT_SPACK_ENV=${PROJECT_SPACK_ENV:-/usr/workspace/xcap/spack-env/${PROJECT_TYPE}/${PROJECT_SPACK_ENV_VERSION}}
  AMP_DATA=${AMP_DATA:-/usr/workspace/xcap/spack-env/AMP-Data.tar.gz}
else
  echo "Unkown system '${SYSTEM_NAME}'"
  false
  return
fi

if [[ "${SPACK_ENV_NAME}" == "custom-spec" ]] && [[ -z "${SPACK_ENV_SPEC}" ]]; then
  echo "Spack environment 'custom-spec' requires SPACK_ENV_SPEC environment variable to be set!"
  false
  return
fi

if [[ "${SPACK_ENV_NAME}" == "custom-file" ]] && [[ -z "${SPACK_ENV_FILE}" ]]; then
  echo "Spack environment 'custom-file' requires SPACK_ENV_FILE environment variable to be set!"
  false
  return
fi

###############################################################################
# Generic steps
###############################################################################

#------------------------------------------------------------------------------
# Prepare Spack instance
#------------------------------------------------------------------------------
prepare_spack() {
  section start "prepare_spack[collapsed=true]" "Prepare Spack"
  umask 0007
  mkdir -p $TMPDIR
  source $PROJECT_SPACK_ENV/replicate.sh $CI_SPACK_ENV
  section end "prepare_spack"
}

#------------------------------------------------------------------------------
# Setup up Spack environment
#------------------------------------------------------------------------------
prepare_env() {
  section start "prepare_env[collapsed=true]" "Create environment"
  echo "Activating ${SPACK_ENV_NAME} environment on ${SYSTEM_NAME}"

  if [[ "${SPACK_ENV_NAME}" == "custom-spec" ]]; then
    SPACK_ENV_FILE=${CI_SPACK_ENV}/systems/${SYSTEM_NAME}/${PROJECT_GROUP}/${PROJECT_NAME}/custom/spack.yaml
  fi

  if [[ "${SPACK_ENV_NAME}" == "custom-spec" ]] || [[ "${SPACK_ENV_NAME}" == "custom-file" ]]; then
    mkdir -p ${CI_SPACK_ENV}/systems/${SYSTEM_NAME}/${PROJECT_GROUP}/${PROJECT_NAME}/${SPACK_ENV_NAME}
    cp ${SPACK_ENV_FILE} ${CI_SPACK_ENV}/systems/${SYSTEM_NAME}/${PROJECT_GROUP}/${PROJECT_NAME}/${SPACK_ENV_NAME}/spack.yaml
  fi

  source ${CI_SPACK_ENV}/systems/${SYSTEM_NAME}/activate.sh ${PROJECT_GROUP}/${PROJECT_NAME}/${SPACK_ENV_NAME}
  if [ -d ${SOURCE_DIR}/spack-repo ]; then
    spack repo add ${SOURCE_DIR}/spack-repo
  fi

  spack develop -b ${BUILD_DIR} -p ${SOURCE_DIR} --no-clone ${PROJECT_NAME}@${PROJECT_DEFAULT_BRANCH}

  if [[ "${SPACK_ENV_NAME}" == "custom-spec" ]]; then
    spack add ${SPACK_ENV_SPEC}
  fi

  spack install --include-build-deps --only dependencies -j $(nproc)
  section end "prepare_env"
}

#------------------------------------------------------------------------------
# Run CMake (configured via Project Spack package and selected Spec)
#------------------------------------------------------------------------------
spack_cmake_configure() {
  section start "spack_cmake_configure[collapsed=true]" "Initial Spack CMake Configure"
  spack install --test root --include-build-deps -u cmake -v ${PROJECT_NAME}
  BUILD_ENV=${BUILD_DIR}/build_env.sh
  spack build-env --dump ${BUILD_ENV} ${PROJECT_NAME}
  section end "spack_cmake_configure"
}

#------------------------------------------------------------------------------
# Build project
#------------------------------------------------------------------------------
cmake_build() {
  section start "cmake_build[collapsed=false]" "CMake Build"
  (
  source ${BUILD_ENV}
  cmake -DCMAKE_VERBOSE_MAKEFILE=off -DAMP_DATA=$AMP_DATA -DCMAKE_INSTALL_PREFIX=$PWD/${BUILD_DIR}/install $@ ${BUILD_DIR}
  if ${BUILD_WITH_CTEST}; then
    ctest -VV -S .gitlab/build_and_test.cmake,Configure,Build,$REPORT_ERRORS
  else
    cmake --build ${BUILD_DIR} --parallel
  fi
  )
  section end "cmake_build"
}

#------------------------------------------------------------------------------
# Run tests
#------------------------------------------------------------------------------
cmake_test() {
  section start "testing[collapsed=false]" "Tests"
  (
  source ${BUILD_ENV}
  export CTEST_OUTPUT_ON_FAILURE=1
  if ${BUILD_WITH_CTEST}; then
    ctest -V -S .gitlab/build_and_test.cmake,Test,$REPORT_ERRORS
  else
    ctest --test-dir ${BUILD_DIR} --output-junit tests.xml
  fi
  )
  section end "testing"
}

#------------------------------------------------------------------------------
# Submit results to CDash
#------------------------------------------------------------------------------
cmake_submit() {
  section start "submit[collapsed=false]" "Submit to CDash"
  (
  source ${BUILD_ENV}
  ctest -V -S .gitlab/build_and_test.cmake,Submit
  )
  section end "submit"
}

#------------------------------------------------------------------------------
# Install project
#------------------------------------------------------------------------------
cmake_install() {
  section start install "Install"
  (
  source ${BUILD_ENV}
  cmake --build ${BUILD_DIR} --target install
  )
  section end install
}

#------------------------------------------------------------------------------
# Enter build environment shell
#------------------------------------------------------------------------------
activate_build_env() {
  spack build-env ${PROJECT_NAME} bash
}

###############################################################################
# Run script
###############################################################################
if ${SHOW_HELP_MESSAGE}; then
  echo "Running on $(hostname)"
  
  if [ ${CI} ]; then
    echo " "
    echo -e "${COLOR_BLUE}######################################################################${COLOR_PLAIN}"
    echo " "
    echo -e "${COLOR_BLUE}To recreate this CI run, follow these steps:${COLOR_PLAIN}"
    echo " "
    echo -e "${COLOR_BLUE}ssh ${CLUSTER}${COLOR_PLAIN}"
    echo -e "${COLOR_BLUE}cd /your/${PROJECT_NAME}/checkout${COLOR_PLAIN}"
    if [[ ! -z "${LLNL_FLUX_SCHEDULER_PARAMETERS}" ]]; then
      echo -e "${COLOR_BLUE}flux alloc ${LLNL_FLUX_SCHEDULER_PARAMETERS}${COLOR_PLAIN}"
    elif [[ ! -z "${LLNL_LSF_SCHEDULER_PARAMETERS}" ]]; then
      echo -e "${COLOR_BLUE}bsub -I ${LLNL_LSF_SCHEDULER_PARAMETERS}${COLOR_PLAIN}"
    else
      echo -e "${COLOR_BLUE}salloc ${SCHEDULER_PARAMETERS}${COLOR_PLAIN}"
    fi
    if [[ ! -z "${SPACK_ENV_SPEC}" ]]; then
      echo -e "${COLOR_BLUE}export SPACK_ENV_SPEC='${SPACK_ENV_SPEC}'${COLOR_PLAIN}"
    fi
    if [[ ! -z "${SPACK_ENV_FILE}" ]]; then
      echo -e "${COLOR_BLUE}export SPACK_ENV_FILE='${SPACK_ENV_FILE}'${COLOR_PLAIN}"
    fi
    echo -e "${COLOR_BLUE}source .gitlab/build_and_test.sh ${CLUSTER} ${SPACK_ENV_NAME}${COLOR_PLAIN}"
    echo " "
    echo -e "${COLOR_BLUE}See 'source .gitlab/build_and_test.sh -h' for more options.${COLOR_PLAIN}"
    echo " "
    echo -e "${COLOR_BLUE}######################################################################${COLOR_PLAIN}"
    echo " "
  fi
fi

prepare_spack
if [[ "${UNTIL}" == "spack" ]]; then return; fi

prepare_env
if [[ "${UNTIL}" == "env" ]]; then return; fi

spack_cmake_configure
if [[ "${UNTIL}" == "cmake" ]]; then return; fi

cmake_build
if [[ "${UNTIL}" == "build" ]]; then return; fi

cmake_test
if [[ "${UNTIL}" == "test" ]]; then return; fi

cmake_install
if [[ "${UNTIL}" == "install" ]]; then return; fi

if ${SUBMIT_TO_CDASH}; then
  cmake_submit
fi
