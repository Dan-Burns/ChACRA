#!/bin/bash
#SBATCH --job-name=chacra
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=chacra_%j.out
#SBATCH --error=chacra_%j.err

module load openmpi
conda activate chacra-env

export PATH="${CONDA_PREFIX}/bin:$PATH"
export OMP_NUM_THREADS=1

# --------------------------------------------------------------------------- #
# Multi-Node & CUDA MPS Settings                                              #
# --------------------------------------------------------------------------- #
MPS_REPLICAS=2
GPUS_PER_NODE=4
N_NODES="${SLURM_NNODES:-4}"
N_GPUS=$(( N_NODES * GPUS_PER_NODE ))                  # 16 GPUs across 4 nodes
N_SYSTEMS=$(( N_GPUS * MPS_REPLICAS ))                 # 32 total replicas
THREAD_PERCENTAGE=$(( 200 / MPS_REPLICAS ))            # e.g. 50% for 4 replicas/GPU

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${THREAD_PERCENTAGE}"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/chacra-mps-${UID}-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY}/logs"

# Launch 1 step per node to manage MPS daemon across all allocated nodes
NODE_STEP=(srun --overlap --exact --mpi=none --nodes="${N_NODES}" --ntasks="${N_NODES}"
    --ntasks-per-node=1 --cpus-per-task=1 --gpu-bind=none --export=ALL)

cleanup() {
    "${NODE_STEP[@]}" bash -c \
    'if [ -s "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" ] &&
    kill -0 "$(cat "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid")" 2>/dev/null; then
    printf "quit\n" | timeout 10 nvidia-cuda-mps-control
    fi' || true
}
trap cleanup EXIT

# Start CUDA MPS daemon on every allocated node
"${NODE_STEP[@]}" bash -c '
set -e 
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
printf "start_server -uid %s\n" "$(id -u)" | 
    timeout 10 nvidia-cuda-mps-control
sleep 2
'

# --------------------------------------------------------------------------- #
# Run HREMD Simulation                                                        #
# --------------------------------------------------------------------------- #
"$CONDA_PREFIX/bin/python" -m chacra.cli run-hremd \
    --system_file [PATH_TO_SYSTEM] \
    --structure_file [PATH_TO_STRUCTURE] \
    --n_cycles 100 \
    --steps_per_cycle 5000 \
    --n_jobs "${N_GPUS}" \
    --mps-replicas "${MPS_REPLICAS}" \
    --n_systems "${N_SYSTEMS}"
