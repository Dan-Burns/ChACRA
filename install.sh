#!/bin/bash
# ============================================================================
# ChACRA Installation Script
#
# Usage:
#   bash install.sh               # fresh install or update existing env
#   bash install.sh --reinstall   # remove existing env and reinstall from scratch
#
# Fast path (no solving): uses conda/explicit-cuda{12,13}.txt if present.
# Fallback: single-stage solve from conda/environment.yaml.
#
# To regenerate spec files after updating environment.yaml:
#   bash tools/generate_locks.sh
# ============================================================================
set -e

# ── 0. Parse flags ────────────────────────────────────────────────────────────
REINSTALL=false
for arg in "$@"; do
    case $arg in
        --reinstall) REINSTALL=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "=== ChACRA Automated Installation ==="

# ── 1. System checks ──────────────────────────────────────────────────────────

# Auto-detect micromamba if not directly on PATH (e.g. defined via MAMBA_EXE or in ~/micromamba)
MICROMAMBA_BIN=""
if command -v micromamba &>/dev/null; then
    MICROMAMBA_BIN="micromamba"
elif [ -n "$MAMBA_EXE" ] && [ -x "$MAMBA_EXE" ]; then
    MICROMAMBA_BIN="$MAMBA_EXE"
elif [ -x "$HOME/micromamba/bin/micromamba" ]; then
    MICROMAMBA_BIN="$HOME/micromamba/bin/micromamba"
elif [ -x "$HOME/.local/bin/micromamba" ]; then
    MICROMAMBA_BIN="$HOME/.local/bin/micromamba"
fi

if ! command -v conda &>/dev/null && ! command -v mamba &>/dev/null && [ -z "$MICROMAMBA_BIN" ]; then
    echo "Error: conda, mamba, or micromamba must be installed."
    exit 1
fi

if ! command -v mpicc &>/dev/null; then
    echo "Error: mpicc not found."
    echo "  e.g.  sudo apt install libopenmpi-dev openmpi-bin"
    exit 1
fi

# ── 2. Detect CUDA ────────────────────────────────────────────────────────────

if [ -n "$CONDA_OVERRIDE_CUDA" ]; then
    CUDA_VER="$CONDA_OVERRIDE_CUDA"
    echo "Using provided CONDA_OVERRIDE_CUDA=$CUDA_VER"
else
    if ! command -v nvidia-smi &>/dev/null; then
        echo "Error: nvidia-smi not found. Ensure NVIDIA drivers are installed or set CONDA_OVERRIDE_CUDA."
        exit 1
    fi
    CUDA_VER=$(nvidia-smi | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
fi

if [ -z "$CUDA_VER" ]; then
    echo "Warning: Could not detect CUDA version."
    CUDA_MAJOR=""
else
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    echo "Target CUDA: $CUDA_VER (major=$CUDA_MAJOR)"
fi

# CuPy wheel selection (12.x and 13.x both use the 12x wheel)
if [ "$CUDA_MAJOR" == "11" ]; then
    CUPY_PKG="cupy-cuda11x"
else
    CUPY_PKG="cupy-cuda12x[ctk]"
fi
echo "Will install: $CUPY_PKG"

# The driver's "CUDA Version" is the newest CUDA runtime it can execute.
# conda-forge happily installs newer CUDA packages (e.g. cuda-nvrtc 13.x on a
# 12.6 driver), which then fail at runtime with
# CUDA_ERROR_UNSUPPORTED_PTX_VERSION.  Cap cuda-version at the driver version.
CUDA_PIN=""
if [ -n "$CUDA_VER" ]; then
    CUDA_PIN="cuda-version<=${CUDA_VER}"
    echo "Will constrain: $CUDA_PIN"
fi

# ver_gt A B  →  true if version A > version B
ver_gt() {
    [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ]
}

# cuda-version recorded in an explicit spec or conda-lock file (empty if none)
_spec_cuda_ver() {
    case "$1" in
        *.txt) grep -Eo '/cuda-version-[0-9]+\.[0-9]+' "$1" | head -1 | grep -Eo '[0-9]+\.[0-9]+' ;;
        *)     awk '/name: cuda-version$/ {f=1; next} f && /version:/ {gsub(/[^0-9.]/, "", $2); print $2; exit}' "$1" ;;
    esac
}

# true if the spec file was built for a CUDA newer than this driver supports
_spec_too_new() {
    local v
    [ -z "$CUDA_VER" ] && return 1
    v=$(_spec_cuda_ver "$1")
    [ -n "$v" ] && ver_gt "$v" "$CUDA_VER"
}

# ── 3. Pick conda frontend ────────────────────────────────────────────────────

# Prefer conda: users activate with it, and mamba/micromamba may keep envs under
# a different root where `conda activate <name>` won't find them.
# Override with e.g.  CONDA_CMD=micromamba ./install.sh
if [ -n "${CONDA_CMD:-}" ]; then
    :
elif command -v conda &>/dev/null; then
    CONDA_CMD="conda"
elif command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="$MICROMAMBA_BIN"
fi
echo "Using: $CONDA_CMD"

# Use the fast libmamba solver even where a site config still selects classic
if [ "$CONDA_CMD" = "conda" ] && conda list -p "$(conda info --base)" conda-libmamba-solver 2>/dev/null \
        | grep -q '^conda-libmamba-solver'; then
    export CONDA_SOLVER=libmamba
fi

ENV_NAME=$(grep -E "^name:" conda/environment.yaml | awk '{print $2}')
ENV_NAME="${ENV_NAME:-chacra-env}"
echo "Environment: $ENV_NAME"

# ── 4. Install the conda environment ─────────────────────────────────────────

# Absolute path of the named environment (empty if it doesn't exist).
# conda prints "name [*] path", micromamba "  name [*] path", so match column 1.
_env_prefix() {
    $CONDA_CMD env list 2>/dev/null | awk -v n="$ENV_NAME" '$1==n {print $NF; exit}'
}

ENV_EXISTS=false
if [ -n "$(_env_prefix)" ]; then
    ENV_EXISTS=true
fi

if [ "$ENV_EXISTS" == "true" ] && [ "$REINSTALL" == "true" ]; then
    echo ""
    echo "--reinstall: removing existing '$ENV_NAME' environment..."
    $CONDA_CMD env remove -n "$ENV_NAME" -y
    ENV_EXISTS=false
fi

if [ "$ENV_EXISTS" == "true" ]; then
    # ── Existing env: just update ──────────────────────────────────────────────
    echo ""
    echo "Environment '$ENV_NAME' already exists — updating from environment.yaml..."
    echo "(To do a clean reinstall from the fast explicit spec, use: bash install.sh --reinstall)"
    $CONDA_CMD env update -n "$ENV_NAME" -f conda/environment.yaml

else
    # ── Fresh install ──────────────────────────────────────────────────────
    # Priority order:
    #   1. @EXPLICIT spec file  (fastest — no solving, no extra tools)
    #   2. conda-lock YAML      (no solving, but needs conda-lock installed)
    #   3. Single-stage env create from environment.yaml

    EXPLICIT_FILE=""
    LOCK_FILE=""

    # 1) Look for @EXPLICIT spec files
    if [ -n "$CUDA_MAJOR" ] && [ -f "conda/explicit-cuda${CUDA_MAJOR}.txt" ]; then
        EXPLICIT_FILE="conda/explicit-cuda${CUDA_MAJOR}.txt"
    elif [ -f "conda/explicit.txt" ]; then
        EXPLICIT_FILE="conda/explicit.txt"
    fi

    # 2) Look for conda-lock YAML files
    if [ -n "$CUDA_MAJOR" ] && [ -f "conda/conda-lock.cuda${CUDA_MAJOR}.yml" ]; then
        LOCK_FILE="conda/conda-lock.cuda${CUDA_MAJOR}.yml"
    elif [ -f "conda/conda-lock.yml" ]; then
        LOCK_FILE="conda/conda-lock.yml"
    fi

    # Reject spec files that pin a CUDA newer than this driver can run
    for _var in EXPLICIT_FILE LOCK_FILE; do
        _f="${!_var}"
        if [ -n "$_f" ] && _spec_too_new "$_f"; then
            echo "Skipping $_f: built for cuda-version $(_spec_cuda_ver "$_f")," \
                 "but this driver supports CUDA $CUDA_VER."
            echo "  (Regenerate with: bash tools/generate_locks.sh)"
            printf -v "$_var" '%s' ""
        fi
    done

    if [ -n "$EXPLICIT_FILE" ]; then
        # ── Explicit spec path (fastest) ───────────────────────────────────
        echo ""
        echo "Found explicit spec: $EXPLICIT_FILE"
        echo "Installing directly (no solving, no extra tools)..."
        $CONDA_CMD create -n "$ENV_NAME" --file "$EXPLICIT_FILE" -y

    elif [ -n "$LOCK_FILE" ]; then
        # ── conda-lock YAML path ──────────────────────────────────────────
        echo ""
        echo "Found lock file: $LOCK_FILE"
        echo "Installing from lock file (no solving required)..."

        if ! command -v conda-lock &>/dev/null; then
            echo "conda-lock not found — installing into base environment..."
            echo "(Tip: generate explicit-*.txt files with 'bash tools/generate_locks.sh'"
            echo " to skip this step on future installs.)"
            $CONDA_CMD install -n base -y -c conda-forge conda-lock
        fi

        conda-lock install -n "$ENV_NAME" "$LOCK_FILE"

    else
        # ── Single-stage solve from environment.yaml ───────────────────────
        echo ""
        echo "No lock or explicit spec files found."
        echo "Creating environment from conda/environment.yaml (single solve)..."
        echo "(To generate spec files for faster installs on other machines:"
        echo "  bash tools/generate_locks.sh)"
        echo ""

        # CONDA_OVERRIDE_CUDA tells the solver which driver it's solving for
        if [ -n "$CUDA_VER" ]; then
            export CONDA_OVERRIDE_CUDA="$CUDA_VER"
            echo "CONDA_OVERRIDE_CUDA=$CONDA_OVERRIDE_CUDA"
        fi

        # environment.yaml is driver-agnostic; add the cuda-version cap for
        # this machine via a temporary copy.
        ENV_FILE="conda/environment.yaml"
        if [ -n "$CUDA_PIN" ]; then
            ENV_FILE=$(mktemp --suffix=.yaml)
            sed "s/^dependencies:\$/dependencies:\n  - \"${CUDA_PIN}\"/" \
                conda/environment.yaml > "$ENV_FILE"
        fi

        $CONDA_CMD env create -n "$ENV_NAME" -f "$ENV_FILE" -y
        [ "$ENV_FILE" != "conda/environment.yaml" ] && rm -f "$ENV_FILE"
    fi
fi

# ── 4a. Resolve the environment's path ──────────────────────────────────────
# Every step below uses this path (never the name), and pip always runs as
# "$ENV_PREFIX/bin/python -m pip", so nothing can land in another environment,
# the system python or ~/.local.

ENV_PREFIX=$(_env_prefix)
PY="$ENV_PREFIX/bin/python"
if [ -z "$ENV_PREFIX" ] || [ ! -x "$PY" ]; then
    echo "Error: could not find python for environment '$ENV_NAME' (looked for $PY)."
    exit 1
fi
# HPC modules often set these, which makes the env's python load another
# install's stdlib / site-packages (and pip install there).
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

PY_PREFIX=$("$PY" -c "import sys; print(sys.prefix)")
if [ "$(cd "$PY_PREFIX" && pwd -P)" != "$(cd "$ENV_PREFIX" && pwd -P)" ]; then
    echo "Error: $PY reports sys.prefix=$PY_PREFIX, not $ENV_PREFIX."
    echo "  Something in your shell redirects python; check: env | grep -E '^PYTHON|^CONDA'"
    exit 1
fi

# ── 4b. Enforce the driver's CUDA cap ──────────────────────────────────────────────────────
# Catches every path above (explicit spec, lock file, env update, solve)
# and existing envs that were built before this check existed.
if [ -n "$CUDA_PIN" ]; then
    INSTALLED_CUDA=$($CONDA_CMD list -p "$ENV_PREFIX" 2>/dev/null \
        | awk '$1=="cuda-version" {print $2; exit}')
    if [ -n "$INSTALLED_CUDA" ] && ver_gt "$INSTALLED_CUDA" "$CUDA_VER"; then
        echo ""
        echo "cuda-version $INSTALLED_CUDA is newer than this driver supports (CUDA $CUDA_VER)."
        echo "Downgrading CUDA packages: $CUDA_PIN ..."
        CONDA_OVERRIDE_CUDA="$CUDA_VER" $CONDA_CMD install -p "$ENV_PREFIX" -y "$CUDA_PIN"
    elif [ -n "$INSTALLED_CUDA" ]; then
        echo "cuda-version $INSTALLED_CUDA is compatible with driver CUDA $CUDA_VER."
    fi
fi

# ── 5. Pip post-install ───────────────────────────────────────────────────────

# conda-lock leaves pip out of the explicit/lock specs (python doesn't depend on
# it). Use python's bundled copy: offline, and no solve that could alter python.
if ! "$PY" -m pip --version &>/dev/null; then
    echo "pip is missing from $ENV_NAME — installing python's bundled pip..."
    "$PY" -m ensurepip
fi

echo ""
echo "Installing pip packages with: $PY"

echo "  Installing $CUPY_PKG..."
"$PY" -m pip install --no-cache-dir "$CUPY_PKG"

echo "  Installing mpi4py (against system MPI: $(which mpicc))..."
"$PY" -m pip install --no-cache-dir mpi4py

# ── Validate MPI ABI ──────────────────────────────────────────────────────
echo ""
echo "  Validating mpi4py installation..."
MPI_CHECK=$("$PY" -c "
try:
    from mpi4py import MPI
    ver = MPI.Get_library_version().strip().split(chr(10))[0]
    print('OK: ' + ver)
except Exception as e:
    print('FAIL: ' + str(e))
" 2>&1 || true)
echo "  $MPI_CHECK"
if echo "$MPI_CHECK" | grep -q "^FAIL"; then
    echo ""
    echo "  WARNING: mpi4py cannot initialise MPI."
    echo "  This usually means mpi4py was built against a different MPI library"
    echo "  than the one currently loaded.  Load the same MPI module that was"
    echo "  active during install, or reinstall:"
    echo "    module load openmpi"
    echo "    $PY -m pip install --no-cache-dir --force-reinstall mpi4py"
    echo ""
fi

# ── Dan-Burns forks: cloned into deps/ and installed in editable mode ────
mkdir -p deps
for repo in femto getcontacts ultracontacts; do
    if [ -d "deps/$repo/.git" ]; then
        echo "  Updating deps/$repo..."
        git -C "deps/$repo" pull --ff-only
    else
        echo "  Cloning $repo (Dan-Burns fork) into deps/$repo..."
        git clone "https://github.com/Dan-Burns/$repo.git" "deps/$repo"
    fi
done

echo "  Installing femto (editable)..."
"$PY" -m pip install --no-cache-dir -e deps/femto

echo "  Installing getcontacts and ultracontacts (editable)..."
"$PY" -m pip install --no-cache-dir --no-deps -e deps/getcontacts -e deps/ultracontacts

echo "  Installing chacra (editable)..."
"$PY" -m pip install --no-cache-dir -e "$(pwd)"

# ── Verify everything resolves inside the environment ────────────────────
echo ""
# find_spec locates packages without importing them: importing cupy-based
# packages needs a GPU, which login nodes don't have.
echo "  Checking packages..."
MISSING=$("$PY" -c "
import importlib.util
names = ['chacra', 'femto', 'getcontacts', 'ultracontacts', 'cupy', 'polars', 'mpi4py']
print(' '.join(n for n in names if importlib.util.find_spec(n) is None))
")
if [ -n "$MISSING" ]; then
    echo "Error: missing from $ENV_NAME: $MISSING"
    exit 1
fi

# `run -n` resolves the name the same way `activate` does
RUN_PREFIX=$($CONDA_CMD run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)" 2>/dev/null || true)
if [ -z "$RUN_PREFIX" ] || [ "$(cd "$RUN_PREFIX" && pwd -P)" != "$(cd "$ENV_PREFIX" && pwd -P)" ]; then
    echo "Error: '$CONDA_CMD activate $ENV_NAME' would use '${RUN_PREFIX:-nothing}', not $ENV_PREFIX."
    echo "  Another environment may share the name; see: $CONDA_CMD env list"
    exit 1
fi
echo "  OK"

# ── Smoke-test OpenMM on CUDA ─────────────────────────────────────────────
# Creating a Context forces kernel compilation, so this catches driver /
# NVRTC mismatches that merely importing openmm would not.
echo ""
echo "  Testing OpenMM CUDA platform..."
CUDA_CHECK=$("$PY" -c "
import openmm
try:
    s = openmm.System(); s.addParticle(1.0)
    openmm.Context(s, openmm.VerletIntegrator(0.001),
                   openmm.Platform.getPlatformByName('CUDA'))
    print('OK')
except Exception as e:
    print('FAIL: ' + str(e).splitlines()[0])
" 2>&1 | tail -1 || true)
echo "  $CUDA_CHECK"
if echo "$CUDA_CHECK" | grep -q "^FAIL"; then
    echo "  WARNING: OpenMM cannot use CUDA on this machine."
    echo "  (Expected on a GPU-less login node; otherwise check the driver"
    echo "   vs. 'conda list cuda-version' in $ENV_NAME.)"
fi

ACTIVATE_CMD="$(basename "$CONDA_CMD")"
echo ""
echo "=== Installation Complete ==="
echo "Environment path:  $ENV_PREFIX   (use it in your batch script)"
echo ""
echo "Activate with:  $ACTIVATE_CMD activate $ENV_NAME"
echo "  (load your modules first; a module loaded after activating can put"
echo "   another python ahead of the environment's)"
echo ""
echo "Notes:"
echo "  • OpenMPI is the recommended MPI implementation for multi-node runs."
echo "    On HPC systems:  module load openmpi"
echo "  • mpi4py is built against whichever MPI was active during install."
echo "    If you switch MPI modules, reinstall mpi4py:"
echo "      $PY -m pip install --no-cache-dir --force-reinstall mpi4py"
echo "  • femto, getcontacts and ultracontacts are editable checkouts in deps/."
echo "    Re-run install.sh (or 'git -C deps/<repo> pull') to update them."
echo ""
echo "Tip: generate a lock file for faster installs on other machines:"
echo "     bash tools/generate_locks.sh"
