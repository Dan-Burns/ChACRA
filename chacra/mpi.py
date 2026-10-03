# -*- coding: utf-8 -*-
"""
MPI launcher utilities for ChACRA.

Provides a single ``build_mpi_command()`` helper that both ``run_hremd`` and
``benchmark_hremd`` use to construct the correct ``mpirun`` / ``srun``
invocation.  Handles:

* auto-detection of the MPI launcher (``mpirun``, ``mpiexec``, ``srun``)
* OpenMPI's ``--oversubscribe`` flag (only added when OpenMPI is detected)
* user-supplied ``--mpi-command`` strings (properly shell-split)
"""

import os
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


def build_mpi_command(
    n_ranks: int,
    *,
    mpi_command: str | None = None,
) -> list[str]:
    """Build the MPI launcher prefix (everything before the program args).

    Parameters
    ----------
    n_ranks : int
        Total number of MPI ranks to launch.
    mpi_command : str or None
        User-supplied MPI launcher string, e.g. ``"srun --mpi=pmix"`` or
        ``"mpirun"``.  If *None*, the launcher is auto-detected from PATH.

    Returns
    -------
    list[str]
        The MPI command tokens, e.g.
        ``["mpirun", "-np", "8", "--oversubscribe"]`` (OpenMPI) or
        ``["srun", "--mpi=pmix", "-n", "8"]``.

    Raises
    ------
    RuntimeError
        If no MPI launcher can be found.
    """
    if mpi_command is not None:
        parts = shlex.split(mpi_command)
        launcher = parts[0]
        extra_flags = parts[1:]
    else:
        launcher = find_mpi_launcher()
        if launcher is None:
            raise RuntimeError(
                "Cannot find mpirun, mpiexec, or srun on PATH.\n"
                "Ensure your MPI module is loaded (e.g. 'module load openmpi')\n"
                "or specify --mpi-command explicitly."
            )
        extra_flags = []

    launcher_basename = os.path.basename(launcher)

    # srun uses -n instead of -np
    if launcher_basename == "srun":
        cmd = [launcher, *extra_flags, "-n", str(n_ranks)]
    else:
        cmd = [launcher, *extra_flags, "-np", str(n_ranks)]
        # OpenMPI requires --oversubscribe when n_ranks > n_physical_cores
        # (common with CUDA MPS).  Other MPI implementations don't have it.
        if _is_openmpi(launcher):
            cmd.append("--oversubscribe")

    return cmd


def configure_mps_env() -> None:
    """Point the CUDA MPS daemon's log directory somewhere user-writable.

    ``nvidia-cuda-mps-control`` logs to ``/var/log/nvidia-mps`` by default,
    which non-root users can't write to (harmless, but it prints a warning
    and loses the daemon logs).  Call this before starting MPS.  An existing
    ``CUDA_MPS_LOG_DIR`` is respected.
    """
    if "CUDA_MPS_LOG_DIR" in os.environ:
        return
    default = "/var/log/nvidia-mps"
    if os.path.isdir(default) and os.access(default, os.W_OK):
        return
    user = os.environ.get("USER") or str(os.getuid())
    log_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"nvidia-mps-log-{user}"
    )
    os.makedirs(log_dir, exist_ok=True)
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
