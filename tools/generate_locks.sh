#!/bin/bash
# ============================================================================
# ChACRA lock-file generator
#
# Run this ONCE on a machine with plenty of RAM (≥64 GB recommended) to
# generate platform+CUDA-specific lock files.  Commit the resulting
# conda-lock.yml files.  Other machines install from the lock file with
# no solving required (see install.sh).
#
# Usage:
#   bash tools/generate_locks.sh              # auto-detect CUDA from driver
#   bash tools/generate_locks.sh 12.6        # specify CUDA version explicitly
# ============================================================================
set -e

CUDA_VER="${1:-}"

# Auto-detect if not provided
if [ -z "$CUDA_VER" ]; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
    if [ -z "$CUDA_VER" ]; then
        echo "Error: Could not detect CUDA version. Pass it explicitly: bash tools/generate_locks.sh 12.6"
        exit 1
    fi
fi

echo "Generating lock file for CUDA ${CUDA_VER} ..."

CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
LOCK_CUDA="$CUDA_VER"

# Install conda-lock into base if not present
if ! command -v conda-lock &>/dev/null; then
    echo "Installing conda-lock into base environment..."
    conda install -n base -y -c conda-forge conda-lock
fi

# Write a temporary virtual-packages spec for this CUDA version
cat > _vp_tmp.yml <<VPEOF
subdirs:
  linux-64:
    packages:
      __cuda: "${LOCK_CUDA}"
VPEOF

LOCK_FILE="conda/conda-lock.cuda${CUDA_MAJOR}.yml"

# The __cuda virtual package alone doesn't stop conda-forge from picking
# newer cuda-nvrtc / cuda-version builds, so cap cuda-version explicitly.
sed "s/^dependencies:\$/dependencies:\n  - \"cuda-version>=${CUDA_MAJOR},<=${LOCK_CUDA}\"/" \
    conda/environment.yaml > _env_tmp.yml

conda-lock \
    -f _env_tmp.yml \
    --virtual-package-spec _vp_tmp.yml \
    --lockfile "$LOCK_FILE" \
    -p linux-64

rm _vp_tmp.yml _env_tmp.yml

# Convert to @EXPLICIT spec file (no conda-lock needed on target machines)
EXPLICIT_FILE="conda/explicit-cuda${CUDA_MAJOR}.txt"
echo ""
echo "Converting to explicit spec file..."
python tools/lock_to_explicit.py "$LOCK_FILE" "$EXPLICIT_FILE"

echo ""
echo "Generated:"
echo "  $LOCK_FILE       (conda-lock YAML — requires conda-lock to install)"
echo "  $EXPLICIT_FILE   (@EXPLICIT spec  — works with plain conda/mamba/micromamba)"
echo ""
echo "Commit both files.  On target machines, install with:"
echo "    bash install.sh"
echo ""
echo "(install.sh will prefer the explicit spec file — no extra tools needed)"

