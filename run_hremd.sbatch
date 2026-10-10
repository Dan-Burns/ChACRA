#!/bin/bash
# ============================================================================
# ChACRA HREMD job script
#
#   sbatch run_hremd.sbatch    # Slurm cluster, any number of nodes
#   bash   run_hremd.sbatch    # single GPU machine, no scheduler
#
# Edit sections 1 and 2 only.  Submit the same script again to continue:
# each submission adds N_CYCLES and resumes from the last checkpoint if a
# run was interrupted.
# ============================================================================

# ── 1. Resources (Slurm only; plain bash ignores these lines) ──────────────
#SBATCH --job-name=chacra
#SBATCH --nodes=1                # number of machines
#SBATCH --gpus-per-node=8        # GPUs per machine
#SBATCH --ntasks-per-node=16     # gpus-per-node x MPS_REPLICAS
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=chacra_%j.log
##SBATCH --partition=gpu         # site specific: delete one '#' to enable
##SBATCH --account=my_account

set -euo pipefail

# ── 2. Settings ─────────────────────────────────────────────────────────────
GPUS_PER_NODE=8      # same as --gpus-per-node
MPS_REPLICAS=2       # replicas running at once on each GPU (1 = off)

CHACRA_ENV="$HOME/micromamba/envs/chacra-env"  # env path from install.sh
PROJECT_DIR="$HOME/chacra_project"             # must be visible to every node

SYSTEM_FILE="system/my_system.xml"
STRUCTURE_FILE="structures/my_minimized.pdb"
N_SYSTEMS=32         # number of replicas (fixed after the first run)
N_CYCLES=1000        # exchange cycles added per submission
SAVE_INTERVAL=10     # save a frame every N cycles

# Leave empty to auto-detect.  Only set it if launching fails, e.g.
#   MPI_COMMAND="srun --mpi=pmix_v4"
#   MPI_COMMAND="mpirun --hostfile hosts.txt --map-by ppr:16:node -x PATH"  # multi-node, no Slurm
MPI_COMMAND=""

# Cluster modules, if your site uses them (must match what install.sh used)
# module load openmpi

# ── Nothing to edit below ───────────────────────────────────────────────────
N_NODES="${SLURM_JOB_NUM_NODES:-1}"
N_GPUS=$(( N_NODES * GPUS_PER_NODE ))

# Use the env's executables directly; no 'activate' needed in batch jobs.
export PATH="$CHACRA_ENV/bin:$PATH"
unset PYTHONHOME PYTHONPATH   # set by some modules; would bypass the env
export PYTHONNOUSERSITE=1

cd "$PROJECT_DIR"

# Stop any of your own MPS daemons left on these nodes by earlier jobs
CLEAN_MPS='for p in /tmp/nvidia-mps /tmp/femto-mps-*/pipe; do [ -e "$p" ] && echo quit | CUDA_MPS_PIPE_DIRECTORY=$p nvidia-cuda-mps-control; done >/dev/null 2>&1; true'
if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --overlap --ntasks-per-node=1 bash -c "$CLEAN_MPS"
else
    bash -c "$CLEAN_MPS"
fi

echo "ChACRA: $N_NODES node(s) x $GPUS_PER_NODE GPU(s) = $N_GPUS GPUs," \
     "$MPS_REPLICAS replica(s)/GPU, $N_SYSTEMS replicas"

chacra run-hremd \
    --system_file "$SYSTEM_FILE" \
    --structure_file "$STRUCTURE_FILE" \
    --n_systems "$N_SYSTEMS" \
    --n_cycles "$N_CYCLES" \
    --save_interval "$SAVE_INTERVAL" \
    --n_jobs "$N_GPUS" \
    --mps-replicas "$MPS_REPLICAS" \
    ${MPI_COMMAND:+--mpi-command "$MPI_COMMAND"}
