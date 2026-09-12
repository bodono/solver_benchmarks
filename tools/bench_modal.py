"""Run campaign shards on Modal: CPU shards on a Linux x86-64 box (where the
SCS wheel's default backend is MKL Pardiso) and GPU shards on an A100 with
an scs-python built against cuDSS.

    modal run tools/bench_modal.py --campaign NAME --spec campaign/campaign.json [--only PATTERN]

Results land on the ``scs-bench-results`` volume under NAME/<shard>/ and
are fetched with ``modal volume get scs-bench-results NAME results/NAME``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("scs-bench")
REPO = Path(__file__).resolve().parents[1]
data_vol = modal.Volume.from_name("scs-bench-data", create_if_missing=True)
results_vol = modal.Volume.from_name("scs-bench-results", create_if_missing=True)

REPO_IGNORE = ["results", "problem_classes", "campaign", ".venv*", ".git", "**/__pycache__", "reports", "*.log"]
SOLVER_PKGS = [
    "osqp", "clarabel", "piqp", "proxsuite", "highspy<1.16", "ortools", "cvxopt", "sdpa-python", "ecos",
]
HARNESS_PKGS = ["click", "h5py", "matplotlib>=3.9", "numpy", "pandas", "pyarrow", "PyYAML", "scipy", "psutil"]

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "libopenblas-dev", "liblapack-dev", "git")
    .pip_install(*HARNESS_PKGS, *SOLVER_PKGS, "scs")
    .add_local_dir(str(REPO), "/root/repo", ignore=REPO_IGNORE, copy=True)
    .run_commands("pip install --no-deps -e /root/repo")
)

CUDA_IMAGE = "nvidia/cuda:12.6.3-devel-ubuntu22.04"
gpu_image = (
    modal.Image.from_registry(CUDA_IMAGE)
    .apt_install("git", "pkg-config", "build-essential", "libopenblas-dev", "liblapack-dev", "ninja-build",
                 "python3", "python3-dev", "python3-pip", "python3-venv", "python-is-python3")
    .run_commands(
        "python -m pip install --upgrade pip",
        # cuDSS from NVIDIA's apt repo; the metapackage installs CUDA 12 and 13 builds, we want 12
        "apt-get update && apt-get install -y cudss",
        "python3 - <<'PY'\n"
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
    .pip_install(*HARNESS_PKGS, *SOLVER_PKGS, "meson-python", "meson", "ninja", "cython")
    .run_commands(
        "git clone --recursive --depth 1 https://github.com/bodono/scs-python.git /opt/scs-python",
        "cd /opt/scs-python && python -m pip install --no-build-isolation "
        "-Csetup-args=-Dlink_cudss=true -Csetup-args=-Dint32=true . 2>&1 | tail -5",
        "python -c 'from scs import _scs_cudss; import scs; print(\"scs\", scs.__version__, \"cuDSS build ok\")'",
    )
    .add_local_dir(str(REPO), "/root/repo", ignore=REPO_IGNORE, copy=True)
    .run_commands("pip install --no-deps -e /root/repo")
)

VOLUMES = {"/data": data_vol, "/results": results_vol}


def _run(campaign: str, name: str, config_yaml: str) -> dict:
    import os, time

    cfg = Path(f"/tmp/{name}.yaml")
    cfg.write_text(config_yaml)
    run_dir = Path("/results") / campaign / name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/data", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "solver_benchmarks.cli", "run", str(cfg), "--run-dir", str(run_dir),
         "--repo-root", "/data", "--no-stream-output"],
        cwd="/root/repo", env=env, capture_output=True, text=True,
    )
    (run_dir / "modal_stdout.log").write_text(proc.stdout[-200_000:])
    (run_dir / "modal_stderr.log").write_text(proc.stderr[-200_000:])
    results_vol.commit()
    rows = 0
    if (run_dir / "results.jsonl").exists():
        rows = sum(1 for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip())
    return {"name": name, "exit": proc.returncode, "rows": rows, "seconds": round(time.time() - t0, 1)}


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4.0, memory=16384, timeout=8 * 3600)
def run_shard_cpu(campaign: str, name: str, config_yaml: str) -> dict:
    return _run(campaign, name, config_yaml)


@app.function(image=gpu_image, volumes=VOLUMES, gpu="A100-80GB", cpu=8.0, memory=32768, timeout=8 * 3600)
def run_shard_gpu(campaign: str, name: str, config_yaml: str) -> dict:
    return _run(campaign, name, config_yaml)


@app.function(image=cpu_image, cpu=2.0)
def probe_cpu() -> str:
    import platform, scs, numpy as np, scipy.sparse as sp
    data = {"A": sp.csc_matrix([[1.0], [-1.0]]), "b": np.array([1.0, 0.0]), "c": np.array([-1.0])}
    info = scs.solve(data, {"l": 2}, verbose=False)["info"]
    return f"{platform.processor() or platform.machine()} scs {scs.__version__} default backend: {info['lin_sys_solver']}"


@app.function(image=gpu_image, volumes=VOLUMES, gpu="A100-80GB", cpu=8.0)
def probe_gpu() -> str:
    import scs, numpy as np, scipy.sparse as sp, subprocess
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    data = {"A": sp.csc_matrix([[1.0], [-1.0]]), "b": np.array([1.0, 0.0]), "c": np.array([-1.0])}
    info = scs.solve(data, {"l": 2}, verbose=False, linear_solver="cudss")["info"]
    return f"{gpu} scs {scs.__version__} backend: {info['lin_sys_solver']}"


@app.local_entrypoint()
def main(campaign: str = "", spec: str = "", only: str = "", probe: bool = False):
    if probe:
        print(probe_cpu.remote()); print(probe_gpu.remote()); return
    shards = json.loads(Path(spec).read_text())
    if only:
        shards = [s for s in shards if only in s["name"]]
    calls = []
    for s in shards:
        fn = run_shard_gpu if s["gpu"] else run_shard_cpu
        calls.append(fn.spawn(campaign, s["name"], Path(s["config"]).read_text()))
    print(f"launched {len(calls)} shards ({sum(1 for s in shards if s['gpu'])} GPU) for campaign {campaign!r}")
    for call in calls:
        r = call.get()
        flag = "" if r["exit"] == 0 else f"  EXIT {r['exit']}"
        print(f"{r['name']:60s} rows={r['rows']:4d} {r['seconds']:8.1f}s{flag}")
