"""Run campaign shards on Modal: CPU shards on a Linux x86-64 box (where the
SCS wheel's default backend is MKL Pardiso) and GPU shards on an A100 with
an scs-python built against cuDSS.

    modal run tools/bench_modal.py --campaign NAME --spec campaign/campaign.json [--only PATTERN]

Results land on the ``scs-bench-results`` volume under NAME/<shard>/ and
are fetched with ``modal volume get scs-bench-results NAME results/NAME``.

Code and data are kept apart. Every image carries the checkout, harness and
reader modules (``problem_classes/*.py``) included, and each shard runs from
it (``--repo-root /root/repo``), so the results' commit is the code that ran.
The ``scs-bench-data`` volume holds data only: it is mounted at ``/data`` and
passed as ``--data-root /data/problem_classes``, so it must hold each dataset
folder the campaign reads under ``problem_classes/``, laid out as in the
repository (``netlib_data/feasible``, ``netlib_data/infeasible`` for the
infeas family, ``kennington``, ``miplib_data``,
``mittelmann``, ``maros_meszaros_data``, ``qplib_data``, ``sdplib_data``,
``mittelmann_sdp_data``, ...). Python files left on the volume by an older
upload are never imported.

Run controls (flags of the entrypoint; without them the caps are the defaults
below and any host will do):

``--threads N``
    OMP, MKL, OpenBLAS and Rayon thread counts of every shard. Default 0: the
    container's reserved CPU count (4 for CPU kinds, 8 for GPU kinds).
``--max-in-flight KIND=N,...``
    Containers of a runner kind at once (defaults in MAX_IN_FLIGHT). A kind
    cannot go above its ceiling in HARD_CAPS, and the GPU kinds together never
    above GPU_LIMIT, the workspace's GPU limit.
``--require-cpu FAMILY/MODEL``
    CPU kinds run only on hosts whose CPU has that family and model (e.g.
    ``25/1``, AMD Zen 3 Milan); a container on another host does nothing and
    the shard is queued again, up to ``--max-host-tries`` distinct containers.
``--require-gpu 'GPU NAME|FAMILY/MODEL'``
    The same for GPU kinds: the GPU name as nvidia-smi gives it and, after the
    bar (optional), the host CPU.

A shard in the spec may also carry ``campaign`` (its results go under that
campaign instead of --campaign), ``require_cpu`` / ``require_gpu`` (in place
of the flags, for that shard) and ``scs_python_ref`` (the run refuses to start
unless the images build that scs-python commit).

The images are fixed before the entrypoint parses its flags, so what goes into
them is set by environment variables on the launching machine:

``SCS_PYTHON_REF=<commit of bodono/scs-python>``
    Build the SCS under test from that commit (CPU images too; see below).
``BENCH_MODAL_PINS=path/to/pins.json``
    Solver and library versions per image, e.g.
    ``{"pins": {"cpu": {"osqp": "1.1.3", "numpy": "2.5.3"}, "cuopt": {"cuopt-cu12": "26.8.0"}}}``.
    Image keys are those of IMAGE_KEYS; ``qtqp_cpu`` and ``cpo3_cpu`` build
    on ``cpu``. A value is a version (``==``) or a specifier (``<1.16``). A
    pin must name a package the image installs. The file may also give
    ``scs_python_ref``.
``QTQP_REPO`` / ``CPO3_REPO``
    Build QTQP / cpo3 from a local checkout instead of PyPI.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path

import modal

app = modal.App("scs-bench")
REPO = Path(__file__).resolve().parents[1]
data_vol = modal.Volume.from_name("scs-bench-data", create_if_missing=True)
results_vol = modal.Volume.from_name("scs-bench-results", create_if_missing=True)

# The dataset folders under problem_classes/ stay out of the images (they live
# on the data volume); the reader modules next to them are shipped.
REPO_IGNORE = [
    "results", "problem_classes/*_data", "problem_classes/kennington", "problem_classes/mittelmann",
    "campaign", ".venv*", ".git", "**/__pycache__", "reports", "*.log",
]
# Same constraint as the pdlp and all extras in pyproject.toml: highspy 1.15.x
# cannot share a process with OR-Tools 9.15.
HIGHSPY = "highspy>=1.0,!=1.15.*"
SOLVER_PKGS = [
    "osqp", "clarabel", "piqp", "proxsuite", HIGHSPY, "ortools", "cvxopt", "sdpa-python", "ecos",
]
HARNESS_PKGS = ["click", "h5py", "matplotlib>=3.9", "numpy", "pandas", "pyarrow", "PyYAML", "scipy", "psutil"]


# --- Image configuration (read on the launching machine) ---------------------
#
# Inside a container these variables are unset; the images are then defined
# without pins, which is harmless (a container never builds an image), so
# nothing here may fail when they are missing.

IMAGE_KEYS = ("cpu", "gpu", "cuopt", "pdlp", "qtqp_gpu")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _req_base(req: str) -> str:
    """The name of a requirement with its extras: 'nvmath-python[cu12]>=1' -> 'nvmath-python[cu12]'."""
    m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[^\]]*\])?", req.strip())
    if not m:
        raise ValueError(f"not a requirement: {req!r}")
    return m.group(0)


def _req_name(req: str) -> str:
    return _norm(_req_base(req).split("[")[0])


def _load_image_config(environ: Mapping[str, str]) -> tuple[str, dict[str, dict[str, str]]]:
    """(scs-python commit to build, {image key: {package: version or specifier}})."""
    ref = _full_sha("SCS_PYTHON_REF", environ.get("SCS_PYTHON_REF"))
    path = (environ.get("BENCH_MODAL_PINS") or "").strip()
    if not path:
        return ref, {}
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"BENCH_MODAL_PINS={path}: {exc}") from exc
    unknown = set(data) - {"scs_python_ref", "pins", "comment"}
    if unknown:
        raise SystemExit(f"BENCH_MODAL_PINS={path}: unknown keys {sorted(unknown)}")
    file_ref = _full_sha(f"{path}: scs_python_ref", data.get("scs_python_ref"))
    if ref and file_ref and ref != file_ref:
        raise SystemExit(f"SCS_PYTHON_REF={ref} but {path} gives scs_python_ref {file_ref}")
    pins: dict[str, dict[str, str]] = {}
    for image, entries in (data.get("pins") or {}).items():
        if image not in IMAGE_KEYS:
            raise SystemExit(f"BENCH_MODAL_PINS={path}: image {image!r} is not one of {IMAGE_KEYS}")
        if not isinstance(entries, dict) or not all(isinstance(v, str) and v.strip() for v in entries.values()):
            raise SystemExit(f"BENCH_MODAL_PINS={path}: pins of {image!r} must map package names to versions")
        pins[image] = {_norm(name): spec.strip() for name, spec in entries.items()}
    # The cpu and gpu images (and the images built on cpu) build SCS from the ref;
    # the pdlp, cuopt and qtqp_gpu images keep PyPI's scs whatever the ref is.
    if (ref or file_ref) and any("scs" in pins.get(image, {}) for image in ("cpu", "gpu")):
        raise SystemExit(f"BENCH_MODAL_PINS={path}: pins scs on an image that builds SCS from scs-python "
                         f"{ref or file_ref}; drop the scs pin")
    return ref or file_ref, pins


def _full_sha(what: str, value) -> str:
    """A full 40-hex commit or empty. ``git fetch origin <sha>`` (the image build)
    fetches only full SHAs, and a branch name would keep the image layer's command
    the same while the branch moves, so Modal's cache would reuse an old build."""
    ref = str(value or "").strip()
    if ref and not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise SystemExit(f"{what}={ref!r}: give the full 40-character commit SHA")
    return ref


SCS_PYTHON_REF, PINS = _load_image_config(os.environ)
_USED_PINS: set[tuple[str, str]] = set()


def _pinned(image: str, reqs: list[str]) -> list[str]:
    """``reqs`` with the pins of ``image`` (an IMAGE_KEYS entry) applied by package name."""
    pins = PINS.get(image, {})
    out = []
    for req in reqs:
        name = _req_name(req)
        if name in pins:
            _USED_PINS.add((image, name))
            spec = pins[name]
            req = _req_base(req) + (spec if spec[0] in "<>=!~" else f"=={spec}")
        out.append(req)
    return out


def _local_checkout(var: str) -> str | None:
    """A local checkout named by ``var``; None when the variable is unset or empty."""
    local = os.environ.get(var)
    if local and not Path(local).is_dir():
        raise SystemExit(f"{var}={local} is not a directory; unset it to install from PyPI")
    return local or None


# --- SCS built from source ------------------------------------------------------
#
# By default the CPU images install the released scs from PyPI and the GPU image
# builds the default branch of scs-python against cuDSS. With SCS_PYTHON_REF
# set, both the CPU image and the GPU image build that commit, with the SCS
# core its scs_source submodule pins. The CPU build is configured like the
# x86-64 Linux wheel on PyPI: the MKL Pardiso backend links Intel's static,
# sequential MKL (the mkl-static release the wheel build uses), so it is the
# default (AUTO) backend, as in a pip install. The other backends use the
# image's OpenBLAS, where the wheel vendors its own.
SCS_PYTHON_URL = "https://github.com/bodono/scs-python.git"
MKL_STATIC_VERSION = "2026.1.0"  # as in scs-python's .github/scripts/install_mkl_static.sh


def _logged_build(cmd: str) -> str:
    """Run a build quietly; on failure print the end of its log and fail the image
    step (piping into tail would hide the error and let a failed build pass)."""
    return f"({cmd}) > /tmp/scs_build.log 2>&1 && tail -3 /tmp/scs_build.log || (tail -200 /tmp/scs_build.log; exit 1)"


def _scs_python_checkout() -> list[str]:
    """Commands that leave the scs-python source to build in /opt/scs-python."""
    if not SCS_PYTHON_REF:
        return [f"git clone --recursive --depth 1 {SCS_PYTHON_URL} /opt/scs-python"]
    return [
        f"git clone {SCS_PYTHON_URL} /opt/scs-python",
        f"cd /opt/scs-python && git fetch origin {SCS_PYTHON_REF} && git checkout --detach {SCS_PYTHON_REF} "
        "&& git submodule update --init --recursive "
        "&& git log --oneline -1 && git -C scs_source log --oneline -1",
    ]


def _with_scs_source(image: modal.Image) -> modal.Image:
    """Replace the PyPI scs in ``image`` by a build of SCS_PYTHON_REF (see above)."""
    if not SCS_PYTHON_REF:
        return image
    return (
        image.apt_install("pkg-config", "ninja-build")
        .pip_install("meson-python", "meson", "ninja", "cython")
        .run_commands(
            *_scs_python_checkout(),
            # Intel's static MKL archives, laid out as scs-python's wheel build does
            "python -m pip download --quiet --no-deps --only-binary=:all: --platform manylinux_2_28_x86_64 "
            f"-d /tmp/mkl mkl-static=={MKL_STATIC_VERSION} mkl-include=={MKL_STATIC_VERSION}",
            "for w in /tmp/mkl/*.whl; do python -m zipfile -e $w /tmp/mkl/unpacked; done "
            "&& mkdir -p /opt/mkl-static && cp -r /tmp/mkl/unpacked/*.data/data/. /opt/mkl-static/ && rm -rf /tmp/mkl",
            "cd /opt/scs-python && " + _logged_build(
                "python -m pip install -v --no-build-isolation --no-deps --force-reinstall "
                "-Csetup-args=-Dmkl_static_prefix=/opt/mkl-static ."),
            "python -c 'import scs, numpy; from scs import _scs_mkl; "
            "print(\"scs\", scs.__version__, \"with MKL; numpy\", numpy.__version__)'",
            "git -C /opt/scs-python/scs_source log -1 --format='scs_source %H %s'",
        )
    )


cpu_image = (
    _with_scs_source(
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential", "libopenblas-dev", "liblapack-dev", "git")
        .pip_install(*_pinned("cpu", [*HARNESS_PKGS, *SOLVER_PKGS, "scs"]))
    )
    .add_local_dir(str(REPO), "/root/repo", ignore=REPO_IGNORE, copy=True)
    .run_commands("pip install --no-deps -e /root/repo")
)

CUDA_IMAGE = "nvidia/cuda:12.6.3-devel-ubuntu22.04"
# Python 3.12 from Modal rather than Ubuntu 22.04's 3.10: the harness needs 3.11
# or newer.
gpu_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("git", "pkg-config", "build-essential", "libopenblas-dev", "liblapack-dev", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade pip",
        # cuDSS from NVIDIA's apt repo; the metapackage installs CUDA 12 and 13 builds, we want 12
        "apt-get update && apt-get install -y cudss",
        "python - <<'PY'\n"
        "import glob, os, pathlib\n"
        "inc = sorted(glob.glob('/usr/include/libcudss/12/cudss.h')) or sorted(glob.glob('/usr/include/**/cudss.h', recursive=True))\n"
        "lib = sorted(glob.glob('/usr/lib/x86_64-linux-gnu/libcudss/12/libcudss.so')) or sorted(glob.glob('/usr/lib/**/libcudss.so', recursive=True))\n"
        "incdir = os.path.dirname(inc[0]); libdir = os.path.dirname(lib[0])\n"
        "pc = 'prefix=/usr\\nincludedir=' + incdir + '\\nlibdir=' + libdir + '\\n\\nName: cudss\\nDescription: NVIDIA cuDSS\\nVersion: 0\\nCflags: -I' + incdir + '\\nLibs: -L' + libdir + ' -lcudss\\n'\n"
        "pathlib.Path('/usr/local/lib/pkgconfig').mkdir(parents=True, exist_ok=True)\n"
        "pathlib.Path('/usr/local/lib/pkgconfig/cudss.pc').write_text(pc)\n"
        "print('cudss.pc ->', incdir, libdir)\n"
        "PY",
    )
    .env({
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PKG_CONFIG_PATH": "/usr/local/cuda/lib64/pkgconfig:/usr/local/cuda/pkgconfig:/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcudss/12:/usr/lib/x86_64-linux-gnu",
    })
    .pip_install(*_pinned("gpu", [*HARNESS_PKGS, *SOLVER_PKGS, "meson-python", "meson", "ninja", "cython"]))
    .run_commands(
        # Modal's standalone Python ships a python-3.12.pc that names its build
        # prefix, /install; with /usr/local/lib/pkgconfig on PKG_CONFIG_PATH (for
        # cudss.pc) Meson takes the headers from there. Point it at the real prefix.
        "[ -e /install ] || ln -s \"$(python -c 'import sys; print(sys.base_prefix)')\" /install",
        "ls /install/include/python3.12/Python.h",
        *_scs_python_checkout(),
        "cd /opt/scs-python && " + _logged_build(
            "python -m pip install -v --no-build-isolation -Csetup-args=-Dlink_cudss=true -Csetup-args=-Dint32=true ."),
        "python -c 'from scs import _scs_cudss; import scs; print(\"scs\", scs.__version__, \"cuDSS build ok\")'",
        "git -C /opt/scs-python/scs_source log -1 --format='scs_source %H %s'",
        "dpkg-query -W 'cudss*' 'libcudss*' || true",
    )
    # NVIDIA cuOpt: GPU PDLP for LP, GPU barrier for QP; the like-for-like GPU comparator.
    .pip_install(*_pinned("gpu", ["cuopt-cu12"]), extra_index_url="https://pypi.nvidia.com")
    # cuopt-cu12 may move numpy: check the SCS build again after it
    .run_commands("python -c 'from scs import _scs_cudss; import scs, numpy; "
                  "print(\"scs\", scs.__version__, \"cuDSS still ok; numpy\", numpy.__version__)'")
    .add_local_dir(str(REPO), "/root/repo", ignore=REPO_IGNORE, copy=True)
    .run_commands("pip install --no-deps -e /root/repo")
)

# cuOpt gets its own image: it bundles its own CUDA 12.9 runtime and cuDSS, and
# the apt cuDSS the SCS build uses shadows them (its barrier QP path then fails
# inside cudssMatrixCreateCsr).
cuopt_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "libopenblas-dev", "liblapack-dev")
    .pip_install(*_pinned("cuopt", [*HARNESS_PKGS, "scs", HIGHSPY, "cuopt-cu12"]),
                 extra_index_url="https://pypi.nvidia.com")
    .add_local_dir(str(REPO), "/root/repo", ignore=REPO_IGNORE, copy=True)
    .run_commands("pip install --no-deps -e /root/repo")
)

# OR-Tools 9.15 and highspy 1.15 cannot be loaded in one Linux process, and
# the harness's MPS reader needs highspy, so the PDLP image takes the same
# highspy constraint as the pdlp extra (tests/test_pdlp_compatibility.py loads
# both in one process and solves an MPS file with PDLP).
pdlp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "libopenblas-dev", "liblapack-dev")
    .pip_install(*_pinned("pdlp", [*HARNESS_PKGS, "ortools==9.15.6755", HIGHSPY, "scs"]))
    .add_local_dir(str(REPO), "/root/repo", ignore=REPO_IGNORE, copy=True)
    .run_commands("pip install --no-deps -e /root/repo")
)

# QTQP on its MKL Pardiso backend and on its cuDSS backend (pip cuDSS + nvmath
# + cupy; no SCS build in that image). The released package is installed from
# PyPI; set QTQP_REPO=/path/to/checkout to build both images from a local
# checkout instead (the SCS 3.3 campaign used a worktree of origin/main). Modal
# resolves every registered image before the entrypoint runs, so nothing here
# may depend on a local path that is not guaranteed to exist.
QTQP_PIN = "qtqp>=0.0.7"
QTQP_IGNORE = ["build", "figures", ".git", "**/__pycache__", "*.tex"]


def _with_qtqp(image: modal.Image, key: str) -> modal.Image:
    local = _local_checkout("QTQP_REPO")
    if local:
        return image.add_local_dir(local, "/root/qtqp", ignore=QTQP_IGNORE, copy=True).run_commands(
            "pip install /root/qtqp"
        )
    return image.pip_install(*_pinned(key, [QTQP_PIN]))


qtqp_cpu_image = _with_qtqp(cpu_image, "cpu").run_commands("python -c 'import qtqp; print(qtqp.LinearSolver.PARDISO)'")
qtqp_gpu_image = (
    _with_qtqp(
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential", "libopenblas-dev", "liblapack-dev")
        .pip_install(*_pinned("qtqp_gpu", [*HARNESS_PKGS, "scs", HIGHSPY, "nvidia-cudss-cu12",
                                           "nvmath-python[cu12]", "cupy-cuda12x"])),
        "qtqp_gpu",
    )
    .add_local_dir(str(REPO), "/root/repo", ignore=REPO_IGNORE, copy=True)
    .run_commands("pip install --no-deps -e /root/repo")
)

# cpo3 (Rust/PyO3 interior-point solver). The released wheel is installed from
# PyPI; set CPO3_REPO=/path/to/checkout to build it inside the image with
# maturin from a local checkout instead. As for QTQP, nothing here may depend on
# a local path that is not guaranteed to exist.
CPO3_PIN = "cpo3>=0.0.1"
CPO3_IGNORE = ["target", ".venv", ".git", "**/__pycache__", "benchmarks", "**/*.so"]


def _with_cpo3(image: modal.Image) -> modal.Image:
    local = _local_checkout("CPO3_REPO")
    if local:
        return (
            image.apt_install("curl")
            .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal", "pip install maturin")
            .add_local_dir(local, "/root/cpo3", ignore=CPO3_IGNORE, copy=True)
            .run_commands("PATH=/root/.cargo/bin:$PATH pip install /root/cpo3")
        )
    return image.pip_install(*_pinned("cpu", [CPO3_PIN]))


cpo3_image = _with_cpo3(cpu_image).run_commands("python -c 'import cpo3; print(cpo3.Settings())'")

_unused_pins = sorted({(image, name) for image, pins in PINS.items() for name in pins} - _USED_PINS)
if _unused_pins:
    raise SystemExit(f"BENCH_MODAL_PINS: no image installs {', '.join(f'{n} ({i})' for i, n in _unused_pins)}")

VOLUMES = {"/data": data_vol, "/results": results_vol}


# --- Runner kinds -----------------------------------------------------------------

# Reserved CPU cores per runner kind (Modal's cpu=); the default thread count.
CPUS = {"cpu": 4, "cpu_big": 4, "pdlp": 4, "qtqp_cpu": 4, "cpo3_cpu": 4, "gpu": 8, "cuopt": 8, "qtqp_gpu": 8}
# Runner kinds on CPU hosts, to which --require-cpu applies, and on GPU hosts,
# to which --require-gpu applies.
CPU_KINDS = {"cpu", "cpu_big", "pdlp", "qtqp_cpu", "cpo3_cpu"}
GPU_KINDS = {"gpu", "cuopt", "qtqp_gpu"}
# Containers of each kind at once, by default (--max-in-flight changes them).
MAX_IN_FLIGHT = {"cpu": 16, "cpu_big": 8, "pdlp": 8, "gpu": 2, "cuopt": 1, "qtqp_cpu": 8, "qtqp_gpu": 2, "cpo3_cpu": 8}
# The most --max-in-flight may ask for, and each function's max_containers:
# 32 CPU and 16 big-CPU containers, 10 GPUs (the workspace's GPU limit; above
# it Modal queues the extra shards while the budget guard charges them from
# spawn). The other 64 GiB CPU kinds stay at their defaults.
HARD_CAPS = {"cpu": 32, "cpu_big": 16, "pdlp": 8, "qtqp_cpu": 8, "cpo3_cpu": 8, "gpu": 10, "cuopt": 10, "qtqp_gpu": 10}
GPU_LIMIT = 10  # GPU containers at once, all GPU kinds together
MAX_HOST_TRIES = 40  # distinct wrong-host containers per shard (one container can reject several)
# A container whose stop did not take effect can return the same requeued shard
# again and again without adding a try; this many wrong-host results per allowed
# try (so 120 at the default 40) ends the shard as failed instead of letting it
# loop until the budget guard trips. Higher tolerates more rejected inputs per
# container; lower fails a shard sooner when containers drain slowly.
WRONG_HOST_RESULTS_PER_TRY = 3
MAX_POLL_ERRORS = 30  # consecutive transient errors polling one call (about 10 minutes)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS")
SHARD_TIMEOUT = 20 * 3600


# --- Inside the container ------------------------------------------------------------

def _cpu_family_model() -> str:
    """'family/model' of the host CPU, e.g. '25/1' for AMD EPYC Zen 3 (Milan). gVisor
    masks the model name string but not these numbers."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return "?/?"
    fam = re.search(r"^cpu family\s*:\s*(\d+)", text, re.M)
    mod = re.search(r"^model\s*:\s*(\d+)", text, re.M)
    return f"{fam.group(1) if fam else '?'}/{mod.group(1) if mod else '?'}"


def _command_output(args: list[str]) -> str:
    """Standard output of a command, or '' (the host record is best effort)."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60, check=False).stdout
    except Exception:
        return ""


def _gpu_name() -> str:
    lines = _command_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip().splitlines()
    return lines[0].strip() if lines else "none"


def _host_ok(require_cpu: str, require_gpu: str) -> tuple[bool, str]:
    """require_cpu 'FAMILY/MODEL'; require_gpu 'GPU NAME|FAMILY/MODEL' (the GPU name,
    e.g. 'NVIDIA A100-SXM4-80GB', must match exactly, and the host CPU too if given)."""
    cpu = _cpu_family_model()
    if require_gpu:
        want_gpu, _, want_cpu = require_gpu.partition("|")
        gpu = _gpu_name()
        return (gpu == want_gpu.strip() and (not want_cpu or cpu == want_cpu.strip())), f"{gpu}|{cpu}"
    if require_cpu:
        return cpu == require_cpu.strip(), cpu
    return True, cpu


# Installed versions written to host.txt (the rows do not record all of them).
HOST_PACKAGES = sorted({"scs", "numpy", "scipy", *(_req_name(r) for r in SOLVER_PKGS), "cuopt-cu12", "qtqp",
                        "cpo3", "nvidia-cudss-cu12", "nvmath-python", "cupy-cuda12x"})


def _host_report() -> str:
    """The host and software of this container, in the lines host.txt records:
    CPU, core count, GPU, cuDSS packages, ``pkg NAME VERSION`` lines and the
    scs-python / scs_source commits of a source build."""
    import importlib.metadata as md

    lines = []
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        cpuinfo = ""
    for pattern in (r"^model name\s*:.*$", r"^cpu family\s*:.*$", r"^model\s*:.*$"):
        m = re.search(pattern, cpuinfo, re.M)
        if m:
            lines.append(m.group(0))
    lines.append(str(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()))
    lines.append(_command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                  "--format=csv,noheader"]).strip())
    lines.append(_command_output(["dpkg-query", "-W", "libcudss*"]).strip())
    for pkg in HOST_PACKAGES:
        try:
            lines.append(f"pkg {pkg} {md.version(pkg)}")
        except md.PackageNotFoundError:
            pass
    commits = [_command_output(["git", "-C", d, "rev-parse", "HEAD"]).strip() or "none"
               for d in ("/opt/scs-python", "/opt/scs-python/scs_source")]
    lines.append(f"scs-python {commits[0]} scs_source {commits[1]}")
    return "".join(line + "\n" for line in lines if line)


def _run(campaign: str, name: str, config_yaml: str, reserved_cpus: int = 4, *, threads: int = 0,
         require_cpu: str = "", require_gpu: str = "") -> dict:
    # Modal places a container on whatever host type is free (AMD Zen 3 and Zen 4,
    # several Intel generations), and they differ in speed. When a host type is
    # required, a container on another host does nothing, stops taking inputs (so
    # the retry gets a new container) and reports the host; the orchestrator
    # queues the shard again.
    if require_cpu or require_gpu:
        ok, host = _host_ok(require_cpu, require_gpu)
        if not ok:
            # Best effort: the stop is a server call; if it fails, Modal may hand the
            # retry to this same container, which costs one more try (the orchestrator
            # caps tries), never a wrong-host row.
            try:
                import modal.experimental
                modal.experimental.stop_fetching_inputs()
            except Exception as exc:
                print(f"stop_fetching_inputs failed: {exc!r}", flush=True)
            return {"name": name, "exit": 75, "rows": 0, "seconds": 0.0, "wrong_host": host,
                    "task": os.environ.get("MODAL_TASK_ID", "")}
    cfg = Path(f"/tmp/{name}.yaml")
    cfg.write_text(config_yaml)
    run_dir = Path("/results") / campaign / name
    run_dir.mkdir(parents=True, exist_ok=True)
    results = run_dir / "results.jsonl"

    def rows_now() -> int:
        return sum(1 for line in results.read_text().splitlines() if line.strip()) if results.exists() else 0

    # Modal's cpu= request is a reservation (containers see and may burst to
    # more cores), so every threading runtime gets the same count: OpenMP, MKL,
    # OpenBLAS and Rayon. The runner leaves variables that are set alone, so
    # its workers see these too.
    count = str(threads or reserved_cpus)
    env = dict(os.environ, **dict.fromkeys(THREAD_VARS, count))
    t0 = time.time()
    # The host of each container that works on the shard, appended: a shard
    # resumed by a later container (a relaunch, a preemption restart) keeps the
    # record of the hosts that solved its earlier rows. The rows do not record
    # the host (gVisor masks the CPU model string), the GPU or cuDSS.
    with (run_dir / "host.txt").open("a") as fh:
        fh.write(f"=== container start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                 f"task={os.environ.get('MODAL_TASK_ID', '')} rows_before={rows_now()} host={_cpu_family_model()}\n")
        fh.write(f"threads {count}\n")
        fh.write(_host_report())
    with (run_dir / "modal_stdout.log").open("w") as out, (run_dir / "modal_stderr.log").open("w") as err:
        # The repository root is the image's checkout: `python -m` puts the
        # working directory first on sys.path, so `bench` imports the shipped
        # solver_benchmarks and problem_classes, and the runner hands its
        # workers those import roots. Only the data comes from the volume.
        proc = subprocess.Popen(
            [sys.executable, "-m", "solver_benchmarks.cli", "run", str(cfg), "--run-dir", str(run_dir),
             "--repo-root", "/root/repo", "--data-root", "/data/problem_classes", "--no-stream-output"],
            cwd="/root/repo", env=env, stdout=out, stderr=err, text=True,
        )
        # Commit the volume every two minutes so that a shard cancelled by the
        # budget guard (or a container limit) resumes from its last result.
        while True:
            try:
                proc.wait(timeout=120)
                break
            except subprocess.TimeoutExpired:
                results_vol.commit()
    with (run_dir / "host.txt").open("a") as fh:
        fh.write(f"=== container end {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                 f"exit={proc.returncode} rows_after={rows_now()}\n")
    results_vol.commit()
    return {"name": name, "exit": proc.returncode, "rows": rows_now(), "seconds": round(time.time() - t0, 1)}


def _run_kind(kind: str, campaign: str, name: str, config_yaml: str, controls: str) -> dict:
    """``controls``: the JSON of the shard's run controls (threads, require_cpu, require_gpu)."""
    return _run(campaign, name, config_yaml, CPUS[kind], **json.loads(controls or "{}"))


def _function(kind: str, image: modal.Image, memory: int, **gpu):
    return app.function(image=image, volumes=VOLUMES, cpu=float(CPUS[kind]), memory=memory,
                        timeout=SHARD_TIMEOUT, max_containers=HARD_CAPS[kind], **gpu)


@_function("cpu", cpu_image, 16384)
def run_shard_cpu(campaign: str, name: str, config_yaml: str, controls: str = "") -> dict:
    return _run_kind("cpu", campaign, name, config_yaml, controls)


# For shards that die at the 16 GiB limit (interior-point solvers on the large
# SDPLIB instances): same image, four times the memory, fewer at a time.
@_function("cpu_big", cpu_image, 65536)
def run_shard_cpu_big(campaign: str, name: str, config_yaml: str, controls: str = "") -> dict:
    return _run_kind("cpu_big", campaign, name, config_yaml, controls)


@_function("gpu", gpu_image, 65536, gpu="A100-80GB")
def run_shard_gpu(campaign: str, name: str, config_yaml: str, controls: str = "") -> dict:
    return _run_kind("gpu", campaign, name, config_yaml, controls)


@_function("cuopt", cuopt_image, 65536, gpu="A100-80GB")
def run_shard_cuopt(campaign: str, name: str, config_yaml: str, controls: str = "") -> dict:
    return _run_kind("cuopt", campaign, name, config_yaml, controls)


@_function("pdlp", pdlp_image, 65536)
def run_shard_pdlp(campaign: str, name: str, config_yaml: str, controls: str = "") -> dict:
    return _run_kind("pdlp", campaign, name, config_yaml, controls)


@_function("qtqp_cpu", qtqp_cpu_image, 65536)
def run_shard_qtqp_cpu(campaign: str, name: str, config_yaml: str, controls: str = "") -> dict:
    return _run_kind("qtqp_cpu", campaign, name, config_yaml, controls)


@_function("qtqp_gpu", qtqp_gpu_image, 65536, gpu="A100-80GB")
def run_shard_qtqp_gpu(campaign: str, name: str, config_yaml: str, controls: str = "") -> dict:
    return _run_kind("qtqp_gpu", campaign, name, config_yaml, controls)


@_function("cpo3_cpu", cpo3_image, 65536)
def run_shard_cpo3_cpu(campaign: str, name: str, config_yaml: str, controls: str = "") -> dict:
    return _run_kind("cpo3_cpu", campaign, name, config_yaml, controls)


@app.function(image=cpu_image, cpu=2.0)
def probe_cpu() -> str:
    import platform

    import numpy as np
    import scipy.sparse as sp
    import scs
    data = {"A": sp.csc_matrix([[1.0], [-1.0]]), "b": np.array([1.0, 0.0]), "c": np.array([-1.0])}
    info = scs.solve(data, {"l": 2}, verbose=False)["info"]
    return f"{platform.processor() or platform.machine()} scs {scs.__version__} default backend: {info['lin_sys_solver']}"


@app.function(image=gpu_image, volumes=VOLUMES, gpu="A100-80GB", cpu=8.0)
def probe_gpu() -> str:
    import subprocess

    import numpy as np
    import scipy.sparse as sp
    import scs
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    data = {"A": sp.csc_matrix([[1.0], [-1.0]]), "b": np.array([1.0, 0.0]), "c": np.array([-1.0])}
    info = scs.solve(data, {"l": 2}, verbose=False, linear_solver="cudss")["info"]
    return f"{gpu} scs {scs.__version__} backend: {info['lin_sys_solver']}"


@app.function(image=cuopt_image, gpu="A100-80GB", cpu=8.0, timeout=900)
def probe_cuopt() -> str:
    """Check the cuOpt API on the installed version: parameter names, the
    termination enum, the QP objective convention and the dual sign."""
    import importlib.metadata as md

    import numpy as np
    import scipy.sparse as sp
    from cuopt.linear_programming import data_model, solver, solver_settings

    from solver_benchmarks.analysis import kkt
    out = [f"cuopt version: {md.version('cuopt-cu12')}"]
    ss = solver_settings.SolverSettings()
    out.append("SolverSettings methods: " + ", ".join(a for a in dir(ss) if not a.startswith('_'))[:700])
    names = None
    for getter in ("get_parameter_names", "parameter_names", "get_all_parameter_names"):
        if hasattr(ss, getter):
            try:
                names = getattr(ss, getter)()
                break
            except Exception as e:
                out.append(f"{getter} failed: {e}")
    mod = solver_settings.solver_settings if hasattr(solver_settings, "solver_settings") else solver_settings
    consts = [a for a in dir(mod) if a.startswith("CUOPT_")]
    out.append(f"param names via getter: {names}")
    for getter in ("toDict", "settings_dict"):
        if hasattr(ss, getter):
            try:
                val = getattr(ss, getter)
                val = val() if callable(val) else val
                out.append(f"{getter}: {val}"[:900])
                break
            except Exception as e:
                out.append(f"{getter} failed: {e}")
    out.append("CUOPT_* constants: " + ", ".join(consts)[:900])
    # min 1/2 x'Px + q'x  s.t. l <= A x <= u, with P = diag(2, 4): solution x = -P^{-1} q if interior
    P = sp.csc_matrix(np.diag([2.0, 4.0]))
    q = np.array([-2.0, -4.0])
    A = sp.csr_matrix(np.eye(2))
    l = np.array([-10.0, -10.0])
    u = np.array([10.0, 10.0])
    for scale, label in [(0.5, "Q = P/2"), (1.0, "Q = P")]:
        m = data_model.DataModel()
        m.set_csr_constraint_matrix(A.data, A.indices.astype(np.int32), A.indptr.astype(np.int32))
        m.set_constraint_lower_bounds(l)
        m.set_constraint_upper_bounds(u)
        m.set_objective_coefficients(q)
        m.set_variable_lower_bounds(np.full(2, -np.inf))
        m.set_variable_upper_bounds(np.full(2, np.inf))
        Q = sp.csr_matrix(P * scale)
        m.set_quadratic_objective_matrix(Q.data, Q.indices.astype(np.int32), Q.indptr.astype(np.int32))
        s = solver_settings.SolverSettings()
        s.set_optimality_tolerance(1e-8)
        sol = solver.Solve(m, s)
        x = np.asarray(sol.get_primal_solution())
        y = np.asarray(sol.get_dual_solution())
        term = sol.get_termination_status()
        out.append(f"{label}: x={np.round(x,4).tolist()} (expect [1,1] for the harness objective) term={term!r} name={getattr(term,'name',None)} obj={sol.get_primal_objective()}")
    # dual sign on an LP with an active constraint: min x s.t. 1 <= x <= 10
    m = data_model.DataModel()
    A = sp.csr_matrix(np.eye(1))
    m.set_csr_constraint_matrix(A.data, A.indices.astype(np.int32), A.indptr.astype(np.int32))
    m.set_constraint_lower_bounds(np.array([1.0]))
    m.set_constraint_upper_bounds(np.array([10.0]))
    m.set_objective_coefficients(np.array([1.0]))
    m.set_variable_lower_bounds(np.array([-np.inf]))
    m.set_variable_upper_bounds(np.array([np.inf]))
    s = solver_settings.SolverSettings()
    s.set_optimality_tolerance(1e-8)
    sol = solver.Solve(m, s)
    x = np.asarray(sol.get_primal_solution())
    y = np.asarray(sol.get_dual_solution())
    r_plus = kkt.qp_residuals(sp.csc_matrix((1,1)), np.array([1.0]), sp.csc_matrix(A), np.array([1.0]), np.array([10.0]), x, y)
    r_minus = kkt.qp_residuals(sp.csc_matrix((1,1)), np.array([1.0]), sp.csc_matrix(A), np.array([1.0]), np.array([10.0]), x, -y)
    out.append(f"LP x={x.tolist()} y={y.tolist()} dual_res(+y)={r_plus['dual_res_rel']:.2e} dual_res(-y)={r_minus['dual_res_rel']:.2e}")
    out.append("termination type: " + str(type(sol.get_termination_status())) + " dir: " + ", ".join(a for a in dir(sol.get_termination_status()) if not a.startswith('_'))[:300])
    return "\n".join(out)


# --- On the launching machine ------------------------------------------------------

# Approximate Modal list prices, USD per container-hour, with a 20% margin on
# top. Check against https://modal.com/pricing; the workspace spending limit
# in the Modal dashboard is the true hard stop, this guard is an estimate.
#   CPU shard: 4 cores x $0.135 + 16 GiB x $0.024  ~ $0.92/h
#   GPU shard: A100-80GB $2.50 + 8 cores x $0.135 + 64 GiB x $0.024 ~ $5.12/h
#   big CPU shard (also the PDLP shard): 4 cores x $0.135 + 64 GiB x $0.024 ~ $2.08/h
RATES_PER_HOUR = {"cpu": 0.92 * 1.2, "cpu_big": 2.08 * 1.2, "pdlp": 2.08 * 1.2, "gpu": 5.12 * 1.2, "cuopt": 5.12 * 1.2,
                  "qtqp_cpu": 2.08 * 1.2, "qtqp_gpu": 5.12 * 1.2, "cpo3_cpu": 2.08 * 1.2}


def _parse_max_in_flight(text: str) -> dict[str, int]:
    """MAX_IN_FLIGHT with the 'KIND=N,...' overrides of --max-in-flight, each within HARD_CAPS."""
    caps = dict(MAX_IN_FLIGHT)
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        kind, sep, value = (s.strip() for s in item.partition("="))
        if not sep or kind not in HARD_CAPS:
            raise SystemExit(f"--max-in-flight: expected KIND=N with KIND one of {sorted(HARD_CAPS)}, got {item!r}")
        try:
            count = int(value)
        except ValueError:
            raise SystemExit(f"--max-in-flight: {item!r} is not KIND=N") from None
        if not 1 <= count <= HARD_CAPS[kind]:
            raise SystemExit(f"--max-in-flight: {kind} must be between 1 and {HARD_CAPS[kind]}, got {count}")
        caps[kind] = count
    return caps


_FAMILY_MODEL = re.compile(r"\d+/\d+")


def _check_host_requirement(name: str, req_cpu: str, req_gpu: str) -> None:
    """Refuse a malformed requirement at launch: a typo would match no host and cost
    max_host_tries cold starts per shard before failing."""
    if req_cpu and not _FAMILY_MODEL.fullmatch(req_cpu.strip()):
        raise SystemExit(f"{name}: require_cpu {req_cpu!r} is not FAMILY/MODEL (e.g. 25/1)")
    if req_gpu:
        gpu, _, cpu = req_gpu.partition("|")
        if not gpu.strip() or (cpu and not _FAMILY_MODEL.fullmatch(cpu.strip())):
            raise SystemExit(f"{name}: require_gpu {req_gpu!r} is not 'GPU NAME' or 'GPU NAME|FAMILY/MODEL'")


def _prepare_shards(shards: list[dict], *, cpu_kind: str = "cpu", big: str = "", threads: int = 0,
                    require_cpu: str = "", require_gpu: str = "", scs_python_ref: str = "") -> list[dict]:
    """Give each shard its runner kind and its run controls; refuse a spec this run cannot honour."""
    if threads < 0:
        raise SystemExit(f"--threads must be 0 (the reserved cores) or more, got {threads}")
    big_set = {n.strip() for n in big.split(",") if n.strip()}
    for s in shards:
        kind = s.get("runner") or ("gpu" if s["gpu"] else "cpu")
        if kind == "cpu" and (cpu_kind != "cpu" or s["name"] in big_set):
            kind = "cpu_big" if s["name"] in big_set else cpu_kind
        if kind not in CPUS:
            raise SystemExit(f"{s['name']}: unknown runner kind {kind!r}")
        s["kind"] = kind
        req_cpu = s.get("require_cpu", require_cpu if kind in CPU_KINDS else "")
        req_gpu = s.get("require_gpu", require_gpu if kind in GPU_KINDS else "")
        if req_gpu and kind not in GPU_KINDS:
            raise SystemExit(f"{s['name']}: require_gpu on a {kind} shard, which runs on a CPU host")
        if req_gpu and req_cpu:
            raise SystemExit(f"{s['name']}: require_cpu on a {kind} shard; give the host CPU in require_gpu "
                             "as 'GPU NAME|FAMILY/MODEL'")
        _check_host_requirement(s["name"], req_cpu, req_gpu)
        want_ref = str(s.get("scs_python_ref") or "")
        if want_ref and not (scs_python_ref and (scs_python_ref.startswith(want_ref) or want_ref.startswith(scs_python_ref))):
            raise SystemExit(f"{s['name']} needs scs-python {want_ref}; the images build "
                             f"{scs_python_ref or 'the PyPI scs'} (set SCS_PYTHON_REF)")
        s["controls"] = {"threads": threads, "require_cpu": req_cpu, "require_gpu": req_gpu}
    return shards


def _orchestrate(shards: list[dict], runners: Mapping, *, campaign: str, budget_usd: float,
                 caps: Mapping[str, int], max_host_tries: int = MAX_HOST_TRIES, poll_seconds: float = 20.0) -> dict:
    """Spawn the shards within the caps and the budget, poll them, queue wrong-host ones again."""
    pending = deque(shards)
    tries: Counter = Counter()  # distinct wrong-host containers per shard
    wrong_results: Counter = Counter()  # every wrong-host result per shard
    seen_tasks: set = set()
    poll_errors: Counter = Counter()  # consecutive transient poll errors per call
    running: dict = {}  # call -> (shard, start)
    spent = 0.0
    done = failed = 0
    print(f"campaign {campaign!r}: {len(shards)} shards, caps {dict(caps)} (GPUs at most {GPU_LIMIT}), "
          f"budget ${budget_usd:.0f}", flush=True)
    while pending or running:
        in_flight = dict.fromkeys(caps, 0)
        for sh, _ in running.values():
            in_flight[sh["kind"]] += 1
        gpus = sum(n for k, n in in_flight.items() if k in GPU_KINDS)
        projected = spent + sum((time.time() - t0) / 3600 * RATES_PER_HOUR[sh["kind"]] for sh, t0 in running.values())
        if projected >= budget_usd:
            print(f"BUDGET REACHED: projected ${projected:.2f} >= ${budget_usd:.0f}; cancelling {len(running)} running, dropping {len(pending)} pending")
            for call in running:
                try:
                    call.cancel()
                except Exception as exc:
                    print(f"cancel failed: {exc}")
            break
        launched = 0
        if pending and not running and all(projected + RATES_PER_HOUR[sh["kind"]] > budget_usd for sh in pending):
            print(f"BUDGET: ${budget_usd - projected:.2f} left cannot fund any pending shard for an hour; dropping {len(pending)} pending", flush=True)
            break
        for _ in range(len(pending)):
            sh = pending[0]
            k = sh["kind"]
            # leave headroom: do not start a shard the budget could not afford for an hour
            if (in_flight[k] >= caps[k] or (k in GPU_KINDS and gpus >= GPU_LIMIT)
                    or projected + RATES_PER_HOUR[k] > budget_usd):
                pending.rotate(-1)
                continue
            pending.popleft()
            call = runners[k].spawn(sh.get("campaign") or campaign, sh["name"], Path(sh["config"]).read_text(),
                                    json.dumps(sh["controls"]))
            running[call] = (sh, time.time())
            in_flight[k] += 1
            gpus += k in GPU_KINDS
            launched += 1
        for call in list(running):
            sh, t0 = running[call]
            try:
                r = call.get(timeout=0)
                poll_errors.pop(call, None)
            except TimeoutError:
                poll_errors.pop(call, None)
                continue
            except Exception as exc:
                # Transient Modal client/API errors (deadline exceeded, connection
                # reset) are not shard failures: the container keeps running.
                if "Deadline" in str(exc) or "Connection" in type(exc).__name__ or "ServiceError" in type(exc).__name__:
                    poll_errors[call] += 1
                    if poll_errors[call] < MAX_POLL_ERRORS:
                        print(f"  transient error polling {sh['name']}: {str(exc)[:80]}; will retry", flush=True)
                        continue
                    # Cancel before giving up, so the container cannot keep running
                    # unseen by the budget guard.
                    try:
                        call.cancel()
                    except Exception as cexc:
                        print(f"cancel failed: {cexc}", flush=True)
                    exc = RuntimeError(f"{MAX_POLL_ERRORS} poll errors in a row (cancelled), last: {str(exc)[:120]}")
                r = {"name": sh["name"], "exit": -1, "rows": 0, "seconds": time.time() - t0, "error": str(exc)[:200]}
            del running[call]
            poll_errors.pop(call, None)
            hours = (time.time() - t0) / 3600
            spent += hours * RATES_PER_HOUR[sh["kind"]]
            if r.get("wrong_host"):
                # A wrong-host container can reject several queued inputs before
                # its stop takes effect: a try counts once per distinct container.
                key = (sh["name"], r.get("task") or f"anon{len(seen_tasks)}")
                if key not in seen_tasks:
                    seen_tasks.add(key)
                    tries[sh["name"]] += 1
                wrong_results[sh["name"]] += 1
                want = sh["controls"]["require_gpu"] or sh["controls"]["require_cpu"]
                # Also bound the results themselves: a wrong-host container whose
                # stop did not take effect would keep taking the requeued shard,
                # adding no new container and so no try, until the budget ran out.
                if tries[sh["name"]] < max_host_tries and wrong_results[sh["name"]] < WRONG_HOST_RESULTS_PER_TRY * max_host_tries:
                    pending.append(sh)
                    print(f"  {sh['name']}: host {r['wrong_host']} (task {r.get('task', '')}), not {want}; "
                          f"queued again (try {tries[sh['name']]})", flush=True)
                    continue
                r["error"] = (f"no {want} host in {max_host_tries} tries" if tries[sh["name"]] >= max_host_tries
                              else f"{wrong_results[sh['name']]} wrong-host results from {tries[sh['name']]} "
                                   "containers (a container kept taking the shard)")
            if r.get("exit") == 0:
                done += 1
            else:
                failed += 1
            flag = "" if r.get("exit") == 0 else f"  EXIT {r.get('exit')} {r.get('error', '')}"
            print(f"{r['name']:58s} rows={r.get('rows', 0):4d} {hours * 60:7.1f} min  spent~${spent:7.2f}{flag}", flush=True)
        if launched or not running:
            print(f"  [{time.strftime('%H:%M')}] running={len(running)} pending={len(pending)} done={done} failed={failed} "
                  f"spent~${spent:.2f} projected~${projected:.2f}", flush=True)
        if pending or running:
            time.sleep(poll_seconds)
    print(f"finished: done={done} failed={failed} pending={len(pending)} spent~${spent:.2f}")
    return {"done": done, "failed": failed, "pending": len(pending), "spent": spent}


@app.local_entrypoint()
def main(
    campaign: str = "",
    spec: str = "",
    only: str = "",
    exclude: str = "",
    names: str = "",
    big: str = "",
    budget_usd: float = 300.0,
    cpu_kind: str = "cpu",
    probe: bool = False,
    probe_cuopt_only: bool = False,
    threads: int = 0,
    max_in_flight: str = "",
    require_cpu: str = "",
    require_gpu: str = "",
    max_host_tries: int = MAX_HOST_TRIES,
):
    if probe:
        print(probe_cpu.remote())
        print(probe_gpu.remote())
        print(probe_cuopt.remote())
        return
    if probe_cuopt_only:
        print(probe_cuopt.remote())
        return
    caps = _parse_max_in_flight(max_in_flight)
    if max_host_tries < 1:
        raise SystemExit(f"--max-host-tries must be at least 1, got {max_host_tries}")
    shards = json.loads(Path(spec).read_text())
    if only:
        shards = [s for s in shards if only in s["name"]]
    if exclude:
        shards = [s for s in shards if exclude not in s["name"]]
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        shards = [s for s in shards if s["name"] in wanted]
    shards = _prepare_shards(shards, cpu_kind=cpu_kind, big=big, threads=threads, require_cpu=require_cpu,
                             require_gpu=require_gpu, scs_python_ref=SCS_PYTHON_REF)
    print(f"scs-python {SCS_PYTHON_REF or '(PyPI scs; GPU image: default branch)'}; "
          f"pins {json.dumps(PINS, sort_keys=True) if PINS else '(none)'}; "
          f"threads {threads or 'reserved cores'}; require_cpu {require_cpu or '(any)'}; "
          f"require_gpu {require_gpu or '(any)'}", flush=True)
    runners = {"cpu": run_shard_cpu, "cpu_big": run_shard_cpu_big, "pdlp": run_shard_pdlp, "gpu": run_shard_gpu,
               "cuopt": run_shard_cuopt, "qtqp_cpu": run_shard_qtqp_cpu, "qtqp_gpu": run_shard_qtqp_gpu,
               "cpo3_cpu": run_shard_cpo3_cpu}
    _orchestrate(shards, runners, campaign=campaign, budget_usd=budget_usd, caps=caps, max_host_tries=max_host_tries)
