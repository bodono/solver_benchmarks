"""Run campaign shards on Modal: CPU shards on a Linux x86-64 box (where the
SCS wheel's default backend is MKL Pardiso) and GPU shards on an A100 with
an scs-python built against cuDSS.

    modal run tools/bench_modal.py --campaign NAME --spec campaign/campaign.json [--only PATTERN]

Results land on the ``scs-bench-results`` volume under NAME/<shard>/ and
are fetched with ``modal volume get scs-bench-results NAME results/NAME``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("scs-bench")
REPO = Path(__file__).resolve().parents[1]
data_vol = modal.Volume.from_name("scs-bench-data", create_if_missing=True)
results_vol = modal.Volume.from_name("scs-bench-results", create_if_missing=True)

REPO_IGNORE = ["results", "problem_classes", "campaign", ".venv*", ".git", "**/__pycache__", "reports", "*.log"]
# The versions the September 2026 campaign ran with (they were what the index
# served then; each row records them in metadata.runtime). The refresh keeps
# the September rows of these solvers on qp, lp and the Mittelmann LPs, and
# re-solves some of them, so the images must not pick up newer releases.
SOLVER_PKGS = [
    "osqp==1.1.3", "clarabel==0.11.1", "piqp==0.6.4", "proxsuite==0.7.3", "highspy==1.15.1", "ortools==9.15.6755",
    "cvxopt==1.3.3", "sdpa-python==0.2.3", "ecos",
]
# The GPU image runs only SCS on cuDSS, so it keeps September's unpinned list.
# highspy reads the MPS files, so it is pinned to the CPU image's version.
SOLVER_PKGS_GPU = ["osqp", "clarabel", "piqp", "proxsuite", "highspy==1.15.1", "ortools", "cvxopt", "sdpa-python", "ecos"]
HARNESS_PKGS = ["click", "h5py", "matplotlib>=3.9", "numpy", "pandas", "pyarrow", "PyYAML", "scipy", "psutil"]
# numpy and scipy as the September merged manifests record them (system.library_versions;
# the rows themselves do not). CPU image only.
CPU_NUMERIC_PINS = ["numpy==2.5.3", "scipy==1.18.1"]

# The SCS under test. By default the CPU images install the released scs from
# PyPI and the GPU image builds the default branch of scs-python against
# cuDSS. To benchmark an SCS that is not released yet, set
# SCS_PYTHON_REF=<commit of bodono/scs-python>: both the CPU image and the GPU
# image then build that commit, with the SCS core its scs_source submodule
# pins. The CPU build is configured like the x86-64 Linux wheel on PyPI: the
# MKL Pardiso backend links Intel's static, sequential MKL (the mkl-static
# release the wheel build uses), so it is the default (AUTO) backend, as in
# a pip install. The other backends use the image's OpenBLAS, where the wheel
# vendors its own. The variable is read on the machine that launches the run;
# inside a container it is unset, and nothing here may then depend on it.
SCS_PYTHON_REF = os.environ.get("SCS_PYTHON_REF")
MKL_STATIC_VERSION = "2026.1.0"  # as in scs-python's .github/scripts/install_mkl_static.sh


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


def _logged_build(cmd: str) -> str:
    """Run a build quietly; on failure print the end of its log and fail the image step.
    (Piping into tail, as before, hid the error and let a failed build pass.)"""
    return f"({cmd}) > /tmp/scs_build.log 2>&1 && tail -3 /tmp/scs_build.log || (tail -200 /tmp/scs_build.log; exit 1)"


def _scs_python_checkout() -> list[str]:
    """Commands that leave the scs-python source to build in /opt/scs-python."""
    if not SCS_PYTHON_REF:
        return ["git clone --recursive --depth 1 https://github.com/bodono/scs-python.git /opt/scs-python"]
    return [
        "git clone https://github.com/bodono/scs-python.git /opt/scs-python",
        f"cd /opt/scs-python && git fetch origin {SCS_PYTHON_REF} && git checkout --detach {SCS_PYTHON_REF} "
        "&& git submodule update --init --recursive "
        "&& git log --oneline -1 && git -C scs_source log --oneline -1",
    ]


cpu_image = (
    _with_scs_source(
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential", "libopenblas-dev", "liblapack-dev", "git")
        .pip_install(*HARNESS_PKGS, *SOLVER_PKGS, *CPU_NUMERIC_PINS, "scs")
    )
    .add_local_dir(str(REPO), "/root/repo", ignore=REPO_IGNORE, copy=True)
    .run_commands("pip install --no-deps -e /root/repo")
)

CUDA_IMAGE = "nvidia/cuda:12.6.3-devel-ubuntu22.04"
# Python 3.12 from Modal rather than Ubuntu 22.04's 3.10: the harness needs 3.11
# or newer since f18e724 (the September image, built before that, had 3.10).
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
    .pip_install(*HARNESS_PKGS, *SOLVER_PKGS_GPU, "meson-python", "meson", "ninja", "cython")
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
    # NVIDIA cuOpt: GPU PDLP for LP and QP, the like-for-like GPU comparator.
    .pip_install("cuopt-cu12", extra_index_url="https://pypi.nvidia.com")
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
    .pip_install(*HARNESS_PKGS, "scs", "highspy<1.16", "cuopt-cu12", extra_index_url="https://pypi.nvidia.com")
    .add_local_dir(str(REPO), "/root/repo", ignore=REPO_IGNORE, copy=True)
    .run_commands("pip install --no-deps -e /root/repo")
)

# OR-Tools 9.15 and highspy 1.15 cannot be loaded in one Linux process, and
# the harness's MPS reader needs highspy, so the PDLP image pins highspy 1.11,
# which was verified to coexist with OR-Tools 9.15 (tools/pair probe).
pdlp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "libopenblas-dev", "liblapack-dev")
    .pip_install(*HARNESS_PKGS, "ortools==9.15.6755", "highspy==1.11.0", "scs")
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


def _with_qtqp(image: modal.Image) -> modal.Image:
    local = os.environ.get("QTQP_REPO")
    if local:
        return image.add_local_dir(local, "/root/qtqp", ignore=QTQP_IGNORE, copy=True).run_commands(
            "pip install /root/qtqp"
        )
    return image.pip_install(QTQP_PIN)


qtqp_cpu_image = _with_qtqp(cpu_image).run_commands("python -c 'import qtqp; print(qtqp.LinearSolver.PARDISO)'")
qtqp_gpu_image = (
    _with_qtqp(
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential", "libopenblas-dev", "liblapack-dev")
        .pip_install(*HARNESS_PKGS, "scs", "highspy<1.16", "nvidia-cudss-cu12", "nvmath-python[cu12]", "cupy-cuda12x")
    )
    .add_local_dir(str(REPO), "/root/repo", ignore=REPO_IGNORE, copy=True)
    .run_commands("pip install --no-deps -e /root/repo")
)

# cpo3 (Rust/PyO3 interior-point solver): built inside the image with maturin from a
# local checkout named by CPO3_REPO (default ~/git/cpo3). The image
# and its runner exist only when that checkout is present, so nothing else depends on it.
CPO3_REPO = Path(os.environ.get("CPO3_REPO", str(Path.home() / "git" / "cpo3")))
CPO3_IGNORE = ["target", ".venv", ".git", "**/__pycache__", "benchmarks", "**/*.so"]
cpo3_image = None
if CPO3_REPO.is_dir():
    cpo3_image = (
        cpu_image
        .apt_install("curl")
        .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal", "pip install maturin")
        .add_local_dir(str(CPO3_REPO), "/root/cpo3", ignore=CPO3_IGNORE, copy=True)
        .run_commands("PATH=/root/.cargo/bin:$PATH pip install /root/cpo3",
                      "python -c 'import cpo3; print(cpo3.Settings())'")
    )

VOLUMES = {"/data": data_vol, "/results": results_vol}


def _cpu_family_model() -> str:
    """'family/model' of the host CPU, e.g. '25/1' for AMD EPYC Zen 3 (Milan). gVisor masks
    the model name string but not these numbers."""
    import re
    text = Path("/proc/cpuinfo").read_text()
    fam = re.search(r"^cpu family\s*:\s*(\d+)", text, re.M)
    mod = re.search(r"^model\s*:\s*(\d+)", text, re.M)
    return f"{fam.group(1) if fam else '?'}/{mod.group(1) if mod else '?'}"


def _run(campaign: str, name: str, config_yaml: str, require_cpu: str = "") -> dict:
    import os
    import time

    # Modal places a container on whatever host type is free (AMD Zen 3 and Zen 4,
    # several Intel generations), and they differ in speed. When a CPU type is
    # required, a container on another host does nothing, stops taking inputs (so
    # the retry gets a new container) and reports the host; the orchestrator
    # queues the shard again.
    if require_cpu:
        host = _cpu_family_model()
        if host != require_cpu:
            import modal.experimental
            modal.experimental.stop_fetching_inputs()
            return {"name": name, "exit": 75, "rows": 0, "seconds": 0.0, "wrong_host": host}
    cfg = Path(f"/tmp/{name}.yaml")
    cfg.write_text(config_yaml)
    run_dir = Path("/results") / campaign / name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Modal's cpu= request is a reservation (containers see and may burst to more cores), so every
    # threading runtime is pinned to the reserved core count: OpenMP, Rayon and the BLAS variants.
    # (Commit 946e11c of the September campaign branch, used there from 2026-09-20.) SCS's MKL
    # backend links sequential MKL, so these pins only affect OpenBLAS users and Rayon.
    threads = os.environ.get("OMP_NUM_THREADS", "4")
    env = dict(os.environ, PYTHONPATH="/data", OMP_NUM_THREADS=threads, RAYON_NUM_THREADS=threads,
               MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
    t0 = time.time()
    # Record the host for this shard: the rows do not (gVisor masks the CPU model
    # string, but family and model numbers show), nor do they record the GPU or cuDSS.
    with (run_dir / "host.txt").open("w") as fh:
        subprocess.run(
            "grep -m1 'model name' /proc/cpuinfo; grep -m1 '^cpu family' /proc/cpuinfo; "
            "grep -m1 -E '^model[[:space:]]+:' /proc/cpuinfo; nproc; "
            "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null; "
            "dpkg-query -W 'libcudss*' 2>/dev/null; "
            "python -c 'import scs, numpy; print(\"scs\", scs.__version__, \"numpy\", numpy.__version__)'",
            shell=True, stdout=fh, stderr=subprocess.STDOUT, cwd="/root/repo", check=False,
        )
    with (run_dir / "modal_stdout.log").open("w") as out, (run_dir / "modal_stderr.log").open("w") as err:
        proc = subprocess.Popen(
            [sys.executable, "-m", "solver_benchmarks.cli", "run", str(cfg), "--run-dir", str(run_dir),
             "--repo-root", "/data", "--no-stream-output"],
            cwd="/root/repo", env=env, stdout=out, stderr=err, text=True,
        )
        # Commit the volume every two minutes so that a shard cancelled by the
        # budget guard (or a container limit) resumes from its last result.
        while proc.poll() is None:
            time.sleep(120)
            results_vol.commit()
    results_vol.commit()
    rows = 0
    if (run_dir / "results.jsonl").exists():
        rows = sum(1 for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip())
    return {"name": name, "exit": proc.returncode, "rows": rows, "seconds": round(time.time() - t0, 1)}


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4.0, memory=16384, timeout=20 * 3600, max_containers=32)
def run_shard_cpu(campaign: str, name: str, config_yaml: str, require_cpu: str = "") -> dict:
    return _run(campaign, name, config_yaml, require_cpu)


# For shards that die at the 16 GiB limit (interior-point solvers on the large
# SDPLIB instances): same image, four times the memory, fewer at a time.
@app.function(image=cpu_image, volumes=VOLUMES, cpu=4.0, memory=65536, timeout=20 * 3600, max_containers=16)
def run_shard_cpu_big(campaign: str, name: str, config_yaml: str, require_cpu: str = "") -> dict:
    return _run(campaign, name, config_yaml, require_cpu)


@app.function(image=gpu_image, volumes=VOLUMES, gpu="A100-80GB", cpu=8.0, memory=65536, timeout=20 * 3600, max_containers=10)
def run_shard_gpu(campaign: str, name: str, config_yaml: str, require_cpu: str = "") -> dict:
    return _run(campaign, name, config_yaml, require_cpu)


@app.function(image=cuopt_image, volumes=VOLUMES, gpu="A100-80GB", cpu=8.0, memory=65536, timeout=20 * 3600, max_containers=1)
def run_shard_cuopt(campaign: str, name: str, config_yaml: str, require_cpu: str = "") -> dict:
    return _run(campaign, name, config_yaml, require_cpu)


@app.function(image=pdlp_image, volumes=VOLUMES, cpu=4.0, memory=65536, timeout=20 * 3600, max_containers=8)
def run_shard_pdlp(campaign: str, name: str, config_yaml: str, require_cpu: str = "") -> dict:
    return _run(campaign, name, config_yaml, require_cpu)


@app.function(image=qtqp_cpu_image, volumes=VOLUMES, cpu=4.0, memory=65536, timeout=20 * 3600, max_containers=8)
def run_shard_qtqp_cpu(campaign: str, name: str, config_yaml: str, require_cpu: str = "") -> dict:
    return _run(campaign, name, config_yaml, require_cpu)


@app.function(image=qtqp_gpu_image, volumes=VOLUMES, gpu="A100-80GB", cpu=8.0, memory=65536, timeout=20 * 3600, max_containers=2)
def run_shard_qtqp_gpu(campaign: str, name: str, config_yaml: str, require_cpu: str = "") -> dict:
    return _run(campaign, name, config_yaml, require_cpu)


if cpo3_image is not None:
    @app.function(image=cpo3_image, volumes=VOLUMES, cpu=4.0, memory=65536, timeout=20 * 3600, max_containers=8)
    def run_shard_cpo3_cpu(campaign: str, name: str, config_yaml: str, require_cpu: str = "") -> dict:
        return _run(campaign, name, config_yaml, require_cpu)


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


# Approximate Modal list prices, USD per container-hour, with a 20% margin on
# top. Check against https://modal.com/pricing; the workspace spending limit
# in the Modal dashboard is the true hard stop, this guard is an estimate.
#   CPU shard: 4 cores x $0.135 + 16 GiB x $0.024  ~ $0.92/h
#   GPU shard: A100-80GB $2.50 + 8 cores x $0.135 + 64 GiB x $0.024 ~ $5.12/h
#   big CPU shard (also the PDLP shard): 4 cores x $0.135 + 64 GiB x $0.024 ~ $2.08/h
RATES_PER_HOUR = {"cpu": 0.92 * 1.2, "cpu_big": 2.08 * 1.2, "pdlp": 2.08 * 1.2, "gpu": 5.12 * 1.2, "cuopt": 5.12 * 1.2,
                  "qtqp_cpu": 2.08 * 1.2, "qtqp_gpu": 5.12 * 1.2, "cpo3_cpu": 2.08 * 1.2}
# The SCS 3.3.2 refresh (Nov 2026) runs 10 A100s, 32 CPU and 16 big-CPU containers at once
# (Brendan, 2026-11-14: "Do it fast"); September ran 2 / 16 / 8. 10 GPUs is the workspace's
# limit on its current plan: above it the extra shards queue, and the guard below would still
# charge them from spawn.
# Runner kinds that run on CPU hosts, to which --require-cpu applies (GPU hosts are not filtered).
CPU_KINDS = {"cpu", "cpu_big", "pdlp", "qtqp_cpu", "cpo3_cpu"}
MAX_HOST_TRIES = 12
MAX_IN_FLIGHT = {"cpu": 32, "cpu_big": 16, "pdlp": 8, "gpu": 10, "cuopt": 1, "qtqp_cpu": 8, "qtqp_gpu": 2, "cpo3_cpu": 8}


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
    require_cpu: str = "",
):
    import time
    from collections import Counter, deque

    if probe:
        print(probe_cpu.remote())
        print(probe_gpu.remote())
        print(probe_cuopt.remote())
        return
    if probe_cuopt_only:
        print(probe_cuopt.remote())
        return
    if campaign.startswith("scs332") and not SCS_PYTHON_REF:
        raise SystemExit("campaign scs332* measures an unreleased SCS: set SCS_PYTHON_REF to the scs-python commit")
    shards = json.loads(Path(spec).read_text())
    if only:
        shards = [s for s in shards if only in s["name"]]
    if exclude:
        shards = [s for s in shards if exclude not in s["name"]]
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        shards = [s for s in shards if s["name"] in wanted]
    big_set = {n.strip() for n in big.split(",") if n.strip()}
    for s in shards:
        s["kind"] = s.get("runner") or ("gpu" if s["gpu"] else "cpu")
        if s["kind"] == "cpu" and (cpu_kind != "cpu" or s["name"] in big_set):
            s["kind"] = "cpu_big" if s["name"] in big_set else cpu_kind
    runners = {"cpu": run_shard_cpu, "cpu_big": run_shard_cpu_big, "pdlp": run_shard_pdlp, "gpu": run_shard_gpu, "cuopt": run_shard_cuopt,
               "qtqp_cpu": run_shard_qtqp_cpu, "qtqp_gpu": run_shard_qtqp_gpu}
    if cpo3_image is not None:
        runners["cpo3_cpu"] = run_shard_cpo3_cpu
    pending = deque(shards)
    tries: Counter = Counter()  # wrong-host attempts per shard (require_cpu)
    running: dict = {}  # call -> (shard, start)
    spent = 0.0
    done = failed = 0
    print(f"campaign {campaign!r}: {len(shards)} shards, caps {MAX_IN_FLIGHT}, budget ${budget_usd:.0f}")
    while pending or running:
        in_flight = {k: sum(1 for (_, (sh, _)) in running.items() if sh["kind"] == k) for k in MAX_IN_FLIGHT}
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
            if in_flight[k] >= MAX_IN_FLIGHT[k] or projected + RATES_PER_HOUR[k] > budget_usd:
                pending.rotate(-1)
                continue
            pending.popleft()
            req = require_cpu if k in CPU_KINDS else ""
            call = runners[k].spawn(campaign, sh["name"], Path(sh["config"]).read_text(), req)
            running[call] = (sh, time.time())
            in_flight[k] += 1
            launched += 1
        for call in list(running):
            sh, t0 = running[call]
            try:
                r = call.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:
                # Transient Modal client/API errors (deadline exceeded, connection
                # reset) are not shard failures: the container keeps running.
                if "Deadline" in str(exc) or "Connection" in type(exc).__name__ or "ServiceError" in type(exc).__name__:
                    print(f"  transient error polling {sh['name']}: {str(exc)[:80]}; will retry", flush=True)
                    continue
                r = {"name": sh["name"], "exit": -1, "rows": 0, "seconds": time.time() - t0, "error": str(exc)[:200]}
            del running[call]
            hours = (time.time() - t0) / 3600
            spent += hours * RATES_PER_HOUR[sh["kind"]]
            if r.get("wrong_host"):
                tries[sh["name"]] += 1
                if tries[sh["name"]] < MAX_HOST_TRIES:
                    pending.append(sh)
                    print(f"  {sh['name']}: host {r['wrong_host']}, not {require_cpu}; queued again (try {tries[sh['name']]})", flush=True)
                    continue
                r["error"] = f"no {require_cpu} host in {MAX_HOST_TRIES} tries"
            if r.get("exit") == 0:
                done += 1
            else:
                failed += 1
            flag = "" if r.get("exit") == 0 else f"  EXIT {r.get('exit')} {r.get('error', '')}"
            print(f"{r['name']:58s} rows={r.get('rows', 0):4d} {hours * 60:7.1f} min  spent~${spent:7.2f}{flag}", flush=True)
        if launched or not running:
            print(f"  [{time.strftime('%H:%M')}] running={len(running)} pending={len(pending)} done={done} failed={failed} spent~${spent:.2f}", flush=True)
        time.sleep(20)
    print(f"finished: done={done} failed={failed} pending={len(pending)} spent~${spent:.2f}")
