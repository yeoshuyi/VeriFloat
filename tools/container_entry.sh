#!/bin/sh
# Runs inside the container started by tools/in_container: installs a
# toolchain with the distribution's package manager, builds the package from
# a copy of /src into a virtual environment, and runs the given command.
set -eu
log() { printf '\n=== %s\n' "$*"; }

log "packages"
if command -v apt-get >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null
    apt-get install -y -qq --no-install-recommends ca-certificates g++ make git cmake ninja-build python3 \
        python3-venv python3-dev libgmp-dev libmpfr-dev libmpc-dev >/dev/null
    # NumPy and gmpy2 from the distribution: PyPI has no wheels for every architecture
    apt-get install -y -qq --no-install-recommends python3-numpy python3-gmpy2 >/dev/null 2>&1 || true
    case "${CXX:-}" in *clang*) apt-get install -y -qq --no-install-recommends clang >/dev/null ;; esac
    case "${CXXFLAGS:-}" in *libc++*) apt-get install -y -qq --no-install-recommends libc++-dev libc++abi-dev >/dev/null ;; esac
elif command -v apk >/dev/null; then
    apk add -q g++ make git cmake samurai python3 python3-dev py3-pip gmp-dev mpfr-dev mpc1-dev linux-headers
    case "${CXX:-}" in *clang*) apk add -q clang ;; esac
elif command -v dnf >/dev/null; then
    py=python3
    grep -q 'release 9' /etc/redhat-release 2>/dev/null && py=python3.12     # EL9: the default python3 is 3.9
    dnf install -y -q gcc-c++ make git cmake $py $py-devel $py-pip gmp-devel mpfr-devel libmpc-devel >/dev/null
    dnf install -y -q ninja-build >/dev/null 2>&1 || true
    case "${CXX:-}" in *clang*) dnf install -y -q clang >/dev/null ;; esac
elif command -v pacman >/dev/null; then
    pacman -Syu --noconfirm --needed gcc make git cmake ninja python >/dev/null
    case "${CXX:-}" in *clang*) pacman -S --noconfirm --needed clang >/dev/null ;; esac
elif command -v zypper >/dev/null; then
    zypper -q -n install gcc-c++ make git cmake ninja python3 python3-devel python3-pip gmp-devel mpfr-devel mpc-devel >/dev/null
else
    echo "no known package manager in this image" >&2; exit 2
fi
[ -n "${CXX:-}" ] || unset CXX
[ -n "${CC:-}" ] || unset CC
[ -n "${CXXFLAGS:-}" ] || unset CXXFLAGS

PY=
for p in python3.14 python3.13 python3.12 python3; do
    if command -v $p >/dev/null && $p -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then PY=$p; break; fi
done
[ -n "$PY" ] || { echo "no Python >= 3.12 in this image" >&2; exit 2; }

log "source copy"
mkdir -p /work && cd /src
tar cf - --exclude=.venv --exclude=build --exclude=.git --exclude=__pycache__ --exclude=sim_build --exclude='*.so' . | tar xf - -C /work
cd /work

log "python environment"
$PY -m venv --system-site-packages /venv
. /venv/bin/activate
if [ -d /cache ]; then
    export PIP_CACHE_DIR=/cache/pip VERIFLOAT_CACHE=/cache/$VF_TAG
fi
pip install -q --disable-pip-version-check -U pip >/dev/null
pip install -q pytest
# The external references: binary wheels only (a missing one skips its tests).
for pkg in numpy gmpy2 ml-dtypes gfloat apytypes; do
    pip install -q -U --only-binary :all: "$pkg" >/dev/null 2>&1 || true     # newer than the distribution's, if any
    python -c "import $(echo $pkg | tr - _)" 2>/dev/null || echo "not available here: $pkg"
done
pip install -q ./archive/python

log "build"
pip install -q . > /tmp/build.log 2>&1 || { tail -40 /tmp/build.log; echo "BUILD FAILED"; exit 3; }
grep -i "warning:" /tmp/build.log | sort | uniq -c | head -20 || true
cd /                                      # test the installed package, not the source directory
python - <<'PY'
import platform, sys, sysconfig, verifloat
from verifloat import _core
libc = " ".join(platform.libc_ver()) or "musl or other"
print(f"platform   {platform.platform()}")
print(f"machine    {platform.machine()}  byteorder {sys.byteorder}  libc {libc}")
print(f"python     {sys.version.split()[0]}  compiler {platform.python_compiler()}")
print(f"extension  CXX={sysconfig.get_config_var('CXX')}  simd available {_core.simd_available()}  in use {_core.simd()}")
PY
${CXX:-c++} --version | head -1

if [ -d /corpus ]; then
    log "golden corpus"
    python /work/tools/golden_corpus verify /corpus
fi

log "run: $*"
cd /work
rm -rf src/verifloat                      # so that `import verifloat` is the installed build
exec "$@"
