![chacra_logo](https://github.com/Dan-Burns/ChACRA/assets/58605062/a030ffbb-0a97-4b33-a968-fab2ec7dbee9)

# ChACRA

## **Ch**emically **A**ccurate **C**ontact **R**esponse **A**nalysis

Created by Dan Burns
https://github.com/Dan-Burns/ChACRA

Tools for identifying energy-sensitive interactions in proteins using contact data from Hamiltonian replica exchange molecular dynamics (HREMD). The energy-sensitive interaction modes (chacras) are the principal components of a protein's contact frequencies across temperature. Chacras reveal functionally critical residue interactions and allosteric communication when distinct structural regions share the same mode.

With ChACRA you can run the full pipeline — simulation, contact calculation, and analysis — with a single command.

---

## Installation

### You need
- An NVIDIA GPU driver (`nvidia-smi` works on the GPU machines)
- OpenMPI (`mpicc` on PATH)
- conda (e.g. Miniforge), mamba, or micromamba

### Install

Run these from a terminal on a machine with internet access.

**On an HPC cluster (login node):**
```bash
module load openmpi miniforge      # use your site's module names
export CONDA_OVERRIDE_CUDA=12.4    # see note 1
git clone https://github.com/Dan-Burns/ChACRA.git
cd ChACRA
./install.sh
```

**On your own GPU machine or a cloud GPU instance:**
```bash
sudo apt install libopenmpi-dev openmpi-bin   # if mpicc is missing
git clone https://github.com/Dan-Burns/ChACRA.git
cd ChACRA
./install.sh
```

**Then, every time you use ChACRA:**
```bash
module load openmpi miniforge      # HPC only, and always before activating
conda activate chacra-env
```

At the end, the installer prints the environment path. Put it in `CHACRA_ENV` in `run_hremd.sbatch`.

### Notes
1. **`CONDA_OVERRIDE_CUDA`** is only needed where there is no GPU (e.g. a login node). Set it to the "CUDA Version" shown at the top right of `nvidia-smi` on a compute node (`srun --gpus=1 nvidia-smi`).
2. **The installer uses `conda`** if it's available. To use another tool, run e.g. `CONDA_CMD=micromamba ./install.sh`, then activate with `micromamba activate chacra-env`.
3. **Starting over:** `./install.sh --reinstall` deletes and rebuilds the environment.
4. **Changing MPI modules** later requires a reinstall (`./install.sh --reinstall`), because mpi4py is built against the MPI that was loaded during install.
5. **Don't load a CUDA module.** The environment brings its own CUDA libraries.
6. On a login node, the installer's last step (OpenMM CUDA test) prints a warning. That's expected.

---

## Quick Start

```bash
mkdir ~/chacra_example && cd ~/chacra_example

# Set up project directory with example structure (1tnf_truncated.pdb)
chacra project --example

# Solvate and create OpenMM system (--fix auto-protonates with pdbfixer)
chacra make-simulation -s structures/1tnf_truncated.pdb --fix --name 1tnf_example

# Run HREMD (4 GPUs, 20 replicas, 1000 exchange cycles)
chacra run-hremd \
    --system_file system/1tnf_example_system.xml \
    --structure_file structures/1tnf_example_minimized.pdb \
    --n_cycles 1000 \
    -j 4 \
    -n 20
```

`chacra run-hremd` automatically calls `chacra process-output` after simulation to generate state trajectories, run contact calculations (GPU-accelerated via `ultracontacts` when available), and produce ChACRA analysis.

### Restarts

Re-run the same `chacra run-hremd` command to continue. A new `run_N/` directory is created for each run and results accumulate across runs. If a run crashes mid-simulation, simply re-run — femto will automatically resume from the last checkpoint.

---

## Output

Results are organized by run:

| Directory | Contents |
|---|---|
| `state_trajectories/run_N/` | Per-state XTC trajectories |
| `contact_output/run_N/` | Per-frame contacts and frequency files |
| `analysis_output/run_N/` | ChACRA plots, `.pml` visualization, `top_chacra_contacts.csv` |
| `analysis_output/latest/` | Symlink to the most recent run |

The `total_contacts.parquet` in each run's analysis reflects accumulated data across all runs. The `.pml` and `.csv` files reflect the combined analysis.

If `chacra process-output` fails partway through, rerun it — it skips completed stages automatically.

---

## Visualization

![chacras](https://github.com/Dan-Burns/ChACRA/assets/58605062/00a98056-bd79-4a3f-95ec-656688838301)

*Projections of contact frequency principal components (chacras). The red mode captures decreasing contact probability with temperature (melting). The blue mode captures contacts that strengthen with temperature — often revealing functionally critical interactions.*

Load your PDB and the `.pml` file into PyMOL to visualize the most sensitive contacts colored by their response pattern:

![IGPS_chacras](https://github.com/Dan-Burns/ChACRA/assets/58605062/a8eb2448-26e5-48e6-a421-6b4cc798ac33)

*IGPS chacras: the fifth chacra (orange) captures allosterically coupled sites; the second chacra (blue) captures interactions critical for activity.*

---

## CLI Reference

All commands are accessed via `chacra <command>`. Run `chacra <command> --help` for full options.

| Command | Description |
|---|---|
| `chacra run-hremd` | Run HREMD simulation + post-processing |
| `chacra process-output` | Process HREMD output (trajectories → contacts → analysis) |
| `chacra make-simulation` | Solvate structure and create OpenMM system |
| `chacra project` | Set up project directory |
| `chacra windowed-freqs` | Compute contact frequencies for frame subsets (convergence analysis) |
| `chacra check-convergence` | Run convergence diagnostics on contact data |
| `chacra benchmark-hremd` | Benchmark HREMD throughput and exchange statistics |
| `chacra sweep-benchmark` | Adaptively find the replica count and `--mps-replicas` value for target exchange rates and best throughput |

---

## Notes

- **Replica count**: 20–40 replicas are typical for systems with 50k–300k particles, targeting ~15–25% exchange rates. You can estimate throughput and exchange rates using sweep-benchmark. See `chacra sweep-benchmark --help` for usage.
- **CUDA MPS (multiple replicas per GPU)**: Use `--mps-replicas 2` (or `-r 2`) to run multiple replicas per GPU simultaneously via NVIDIA CUDA MPS. This increases throughput (up to a point) when you have more replicas than GPUs. benchmark-hremd will help you find the optimal number of simultaneous replicas per GPU for your system. Each GPU will process n mps-replicas simultaneously for a cycle and then consume the next n until all replicas are processed.
- **Multi-node runs**: Set `--n_jobs` to the total GPU count across all nodes (e.g. 2 nodes × 4 GPUs = 8) and run `chacra run-hremd` from inside your Slurm allocation. ChACRA launches with `srun`, picks the best MPI plugin your site supports (`srun --mpi=list`), and spreads ranks evenly across the nodes. Each node starts and stops its own CUDA MPS daemon when `--mps-replicas` > 1. If your site needs a specific launcher, pass it once with e.g. `--mpi-command "srun --mpi=pmix_v4"`. It is saved to `chacra_run.json` for later runs.
- **Ligands**: Systems with ligands may require adding custom force names to femto's `_SUPPORTED_FORCES` list in `femto/md/rest.py`. HREMD scaling is limited to the protein.
- **NaN errors**: Usually caused by inadequately minimized starting coordinates.
- **CUDA PTX errors**: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` means conda's CUDA toolkit is newer than your driver supports. The install script handles this automatically.

---

## Citations

Please cite the following if you use ChACRA:

1. Burns, D., Singh, A., Venditti, V. & Potoyan, D. A. Temperature-sensitive contacts in disordered loops tune enzyme I activity. *Proc. Natl. Acad. Sci. U. S. A.* **119**, e2210537119 (2022)

2. Burns, D., Venditti, V. & Potoyan, D. A. Temperature sensitive contact modes allosterically gate TRPV3. *PLoS Comput. Biol.* **19**, e1011545 (2023)

3. Burns, D., Venditti, V. & Potoyan, D. A. Illuminating protein allostery by chemically accurate contact response analysis (ChACRA). *J. Chem. Theory Comput.* (2024)

4. A. Singh,D. Burns,S.L. Sedinkin,S. Das,D.A. Potoyan, & V. Venditti, Integrating NMR and contact-response analysis reveals the allosteric network driving domain closure in Enzyme I, *Proc. Natl. Acad. Sci. U.S.A. **123** e2612191123 (2026)
