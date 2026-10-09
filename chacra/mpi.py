# -*- coding: utf-8 -*-
"""
MPI launcher utilities for ChACRA.

Provides a single ``build_mpi_command()`` helper that both ``run_hremd`` and
``benchmark_hremd`` use to construct the correct ``mpirun`` / ``srun``
invocation.  Handles:

* Slurm allocations: ``srun`` with the best available ``--mpi=`` plugin and
  ranks spread evenly over every allocated node
* auto-detection of the MPI launcher (``mpirun``, ``mpiexec``, ``srun``)
* OpenMPI's ``--oversubscribe`` flag (only added when OpenMPI is detected)
* user-supplied ``--mpi-command`` strings (properly shell-split) as a site
  override

CUDA MPS is managed per node by the MPI ranks themselves
(``femto.md.utils.mpi.node_mps``), so nothing MPS related is needed here.
"""

import os
import re
import shlex
import shutil
import subprocess


def _is_openmpi(launcher: str) -> bool:
    """Return True if *launcher* appears to be an OpenMPI binary."""
    try:
        result = subprocess.run(
            [launcher, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout + result.stderr
        return "open mpi" in output.lower() or "openmpi" in output.lower()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def find_mpi_launcher() -> str | None:
    """Return the first usable MPI launcher on PATH.

    Checks for ``mpirun``, ``mpiexec``, and ``srun`` in that order.
    Returns the absolute path (via ``shutil.which``) or *None*.
    """
    for name in ("mpirun", "mpiexec", "srun"):
        path = shutil.which(name)
        if path is not None:
            return path
    return None


def _srun_mpi_plugin(srun: str) -> str | None:
    """Return the best ``srun --mpi=`` plugin this site supports (pmix > pmi2)."""
    try:
        result = subprocess.run(
            [srun, "--mpi=list"], capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    plugins = set(re.findall(r"\b(pmix(?:_v\d+)?|pmi2)\b", result.stdout + result.stderr))
    pmix_versions = sorted(p for p in plugins if p.startswith("pmix_v"))
    if "pmix" in plugins:
        return "pmix"
    if pmix_versions:
        return pmix_versions[-1]
    return "pmi2" if "pmi2" in plugins else None


def build_mpi_command(
    n_ranks: int,
    *,
    mpi_command: str | None = None,
) -> list[str]:
    """Build the MPI launcher prefix (everything before the program args).

    Inside a Slurm allocation ``srun`` is used with the best MPI plugin the
    site supports, and ranks are spread evenly over all allocated nodes.
    Elsewhere ``mpirun`` / ``mpiexec`` is used.

    Parameters
    ----------
    n_ranks : int
        Total number of MPI ranks to launch (across all nodes).
    mpi_command : str or None
        User-supplied MPI launcher string that overrides auto-detection, e.g.
        ``"srun --mpi=pmix_v4"`` or ``"mpirun"``.

    Returns
    -------
    list[str]
        The MPI command tokens, e.g.
        ``["srun", "--mpi=pmix", "-n", "8", "--ntasks-per-node=4"]`` or
        ``["mpirun", "-np", "8", "--oversubscribe"]`` (OpenMPI).

    Raises
    ------
    RuntimeError
        If no MPI launcher can be found.
    """
    n_nodes = os.environ.get("SLURM_JOB_NUM_NODES") or os.environ.get("SLURM_NNODES")
    n_nodes = int(n_nodes) if n_nodes else None

    if mpi_command is not None:
        parts = shlex.split(mpi_command)
    elif n_nodes is not None and shutil.which("srun") is not None:
        srun = shutil.which("srun")
        plugin = _srun_mpi_plugin(srun)
        parts = [srun] + ([f"--mpi={plugin}"] if plugin else [])
    else:
        launcher = find_mpi_launcher()
        if launcher is None:
            raise RuntimeError(
                "Cannot find mpirun, mpiexec, or srun on PATH.\n"
                "Ensure your MPI module is loaded (e.g. 'module load openmpi')\n"
                "or specify --mpi-command explicitly."
            )
        parts = [launcher]

    launcher = parts[0]
    ranks_per_node = -(-n_ranks // n_nodes) if n_nodes else None

    if os.path.basename(launcher) == "srun":
        cmd = [*parts, "-n", str(n_ranks)]
        if ranks_per_node and not any(p.startswith("--ntasks-per-node") for p in parts):
            cmd.append(f"--ntasks-per-node={ranks_per_node}")
    else:
        cmd = [*parts, "-np", str(n_ranks)]
        # OpenMPI requires --oversubscribe when n_ranks > n_physical_cores
        # (common with CUDA MPS).  Other MPI implementations don't have it.
        if _is_openmpi(launcher):
            cmd.append("--oversubscribe")
            # by default OpenMPI fills the first node's slots before moving on
            if ranks_per_node and "--map-by" not in parts:
                cmd += ["--map-by", f"ppr:{ranks_per_node}:node"]

    return cmd


def configure_mps_env() -> None:
    """Point the CUDA MPS daemon's log directory somewhere user-writable.

    ``nvidia-cuda-mps-control`` logs to ``/var/log/nvidia-mps`` by default,
    which non-root users can't write to (harmless, but it prints a warning
    and loses the daemon logs).  Call this before starting MPS.
    """
    if "CUDA_MPS_LOG_DIRECTORY" in os.environ:
        return
    default = "/var/log/nvidia-mps"
    if os.path.isdir(default) and os.access(default, os.W_OK):
        return
    user = os.environ.get("USER") or str(os.getuid())
    log_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"nvidia-mps-log-{user}"
    )
    os.makedirs(log_dir, exist_ok=True)
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = log_dir
    os.environ["CUDA_MPS_LOG_DIR"] = log_dir


def validate_mpi_install() -> dict:
    """Validate that mpi4py can load and report which MPI library it uses.

    Returns a dict with keys:
        ok (bool): True if mpi4py imported and MPI_Init succeeded
        mpi_library (str): path to the linked libmpi.so (if detectable)
        mpi_version (str): MPI implementation version string
        error (str): error message if ok is False
    """
    result = {"ok": False, "mpi_library": "", "mpi_version": "", "error": ""}
    try:
        from mpi4py import MPI
        result["mpi_version"] = MPI.Get_library_version().strip()
        result["ok"] = True
    except ImportError as e:
        result["error"] = f"mpi4py import failed: {e}"
    except Exception as e:
        result["error"] = f"MPI init failed: {e}"

    # Try to find the linked libmpi
    try:
        import ctypes.util
        lib = ctypes.util.find_library("mpi")
        if lib:
            result["mpi_library"] = lib
    except Exception:
        pass

    return result
