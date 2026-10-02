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

if ! command -v conda &>/dev/null && ! command -v mamba &>/dev/null && ! command -v micromamba &>/dev/null; then
    echo "Error: conda, mamba, or micromamba must be installed."
    exit 1
fi

if ! command -v mpicc &>/dev/null; then
    echo "Error: mpicc not found."
    echo "  e.g.  sudo apt install libopenmpi-dev openmpi-bin"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    exit 1
fi

# ── 2. Detect CUDA ────────────────────────────────────────────────────────────

CUDA_VER=$(nvidia-smi | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
if [ -z "$CUDA_VER" ]; then
    echo "Warning: Could not detect CUDA version."
    CUDA_MAJOR=""
else
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    echo "Detected CUDA: $CUDA_VER (major=$CUDA_MAJOR)"
fi

# CuPy wheel selection (12.x and 13.x both use the 12x wheel)
if [ "$CUDA_MAJOR" == "11" ]; then
    CUPY_PKG="cupy-cuda11x"
else
    CUPY_PKG="cupy-cuda12x"
fi
echo "Will install: $CUPY_PKG"

# ── 3. Pick conda frontend ────────────────────────────────────────────────────

if command -v micromamba &>/dev/null; then
    CONDA_CMD="micromamba"
elif command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
    # Enable libmamba solver — the classic solver OOMs on complex environments
    if ! conda config --show solver 2>/dev/null | grep -q "libmamba"; then
        echo "Enabling libmamba solver..."
        conda install -n base -y conda-libmamba-solver 2>/dev/null || true
        conda config --set solver libmamba 2>/dev/null || true
    fi
fi
echo "Using: $CONDA_CMD"

ENV_NAME=$(grep -E "^name:" conda/environment.yaml | awk '{print $2}')
ENV_NAME="${ENV_NAME:-chacra-env}"
echo "Environment: $ENV_NAME"

# ── 4. Install the conda environment ─────────────────────────────────────────

# Check if environment already exists
ENV_EXISTS=false
if $CONDA_CMD env list 2>/dev/null | grep -qE "^${ENV_NAME}[[:space:]]"; then
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

        # CONDA_OVERRIDE_CUDA helps the solver pick the right CUDA builds
        if [ -n "$CUDA_VER" ]; then
            if [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
                export CONDA_OVERRIDE_CUDA="12.8"
            else
                export CONDA_OVERRIDE_CUDA="$CUDA_VER"
            fi
            echo "CONDA_OVERRIDE_CUDA=$CONDA_OVERRIDE_CUDA"
        fi

        $CONDA_CMD env create -f conda/environment.yaml -y
    fi
fi

# ── 5. Pip post-install ───────────────────────────────────────────────────────

echo ""
echo "Installing pip packages: $CUPY_PKG, mpi4py, femto, getcontacts, ultracontacts, chacra..."

# --------------------------------------------------------------------------
# We need to activate the environment so pip installs into the RIGHT
# site-packages.  `$CONDA_CMD run` does NOT reliably set PATH / PYTHONPATH
# on all HPC setups (especially when mamba is loaded via `module load`).
#
# Instead, we source the conda/mamba/micromamba shell init and activate
# properly inside a subshell.
# --------------------------------------------------------------------------

# Locate the conda init script
_find_conda_init() {
    # micromamba: shell hook
    if [ "$CONDA_CMD" = "micromamba" ]; then
        echo "micromamba"
        return
    fi
    # mamba/conda: look for the shell init script
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || mamba info --base 2>/dev/null || true)"
    if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
        echo "$conda_base/etc/profile.d/conda.sh"
        return
    fi
    # Check CONDA_EXE parent
    if [ -n "$CONDA_EXE" ]; then
        local d
        d="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
        [ -f "$d" ] && echo "$d" && return
    fi
    echo ""
}

CONDA_INIT=$(_find_conda_init)

(
    # Subshell: activate the environment and run pip installs
    set -e

    if [ "$CONDA_CMD" = "micromamba" ]; then
        eval "$(micromamba shell hook --shell bash)"
        micromamba activate "$ENV_NAME"
    elif [ -n "$CONDA_INIT" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_INIT"
        # Also source mamba init if available (for `mamba activate`)
        local_mamba_init="$(dirname "$CONDA_INIT")/mamba.sh"
        [ -f "$local_mamba_init" ] && source "$local_mamba_init"
        conda activate "$ENV_NAME"
    else
        echo "Warning: Could not find conda init script."
        echo "Falling back to '$CONDA_CMD run' (may mis-target pip installs on some HPC systems)."
        # Fall back to CONDA_CMD run — set a flag so the commands below
        # invoke pip via $CONDA_CMD run instead of directly
        export _USE_CONDA_RUN=1
    fi

    _pip() {
        if [ "${_USE_CONDA_RUN:-0}" = "1" ]; then
            $CONDA_CMD run -n "$ENV_NAME" python -m pip "$@"
        else
            python -m pip "$@"
        fi
    }

    # Verify pip is targeting the correct environment
    PIP_TARGET=$(_pip show pip 2>/dev/null | grep "^Location:" | awk '{print $2}' || true)
    if [ -n "$PIP_TARGET" ]; then
        echo "  pip site-packages: $PIP_TARGET"
        if ! echo "$PIP_TARGET" | grep -q "$ENV_NAME"; then
            echo "WARNING: pip target does not contain '$ENV_NAME'."
            echo "         Packages may install into the wrong environment."
            echo "         Consider using: bash install.sh --reinstall"
        fi
    fi

    echo "  Installing $CUPY_PKG..."
    _pip install --no-cache-dir "$CUPY_PKG"

    echo "  Installing mpi4py (against system MPI: $(which mpicc))..."
    _pip install --no-cache-dir mpi4py

    echo "  Installing femto (Dan-Burns fork)..."
    _pip install --no-cache-dir "git+https://github.com/Dan-Burns/femto.git"

    echo "  Installing getcontacts (Dan-Burns fork)..."
    _pip install --no-cache-dir --no-deps "git+https://github.com/Dan-Burns/getcontacts.git"

    echo "  Installing ultracontacts (Dan-Burns fork)..."
    _pip install --no-cache-dir --no-deps "git+https://github.com/Dan-Burns/ultracontacts.git"

    echo "  Installing chacra (editable)..."
    _pip install --no-cache-dir -e "$(pwd)"
)

echo ""
echo "=== Installation Complete ==="
echo "Activate with:  conda activate $ENV_NAME"
echo ""
echo "Tip: generate a lock file for faster installs on other machines:"
echo "     bash tools/generate_locks.sh"
