# ASR CUDA kernels

This folder contains our completed CUDA kernels for the supplied ASR model and the logs used to check and profile them. The files are a supplement to the course repository, not a standalone program. The supplied framework and Python model files are not duplicated here.

## 1. Prerequisites

Clone the course repository and complete the ASR setup described in the [course ASR README](../../hw1-asr/README.md). This includes creating the `mls-asr` environment and downloading the model on a head node with internet access.

For compilation and testing, request a GPU shell from the cluster head node:

```bash
srun -p Teaching --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
```

The saved measurements in this submission were made on an NVIDIA GeForce RTX 2080 Ti (compute capability 7.5) with CUDA 12.8. Timings can differ on other GPUs or under different system load.

## 2. Submitted files

Copy these five files into the matching `hw1-asr/csrc/` directory. They implement the four ASR tasks:

```text
asr/
├── csrc/
│   ├── task1_rope.cu
│   ├── task2_reduce.cuh
│   ├── task2_reduce.cu
│   ├── task3_gemm.cu
│   └── task4_attention.cu
└── output-logs/
    ├── asr-make-test.log
    ├── asr-profile.log
    └── gpu.txt
```

- **RoPE** rotates query and key value pairs based on token position.
- **Reductions** provide warp/block sums and maxima, RMSNorm, LayerNorm, and softmax.
- **Tiled GEMM** implements the model's linear layers, with GELU fused into the relevant epilogue.
- **Attention** computes attention by streaming tiles and maintaining an online softmax summary.

## 3. Apply the submitted kernels

From the course repository root, copy the submitted kernel files into the supplied ASR project:

```bash
cp submission/asr/csrc/task1_rope.cu hw1-asr/csrc/
cp submission/asr/csrc/task2_reduce.cuh hw1-asr/csrc/
cp submission/asr/csrc/task2_reduce.cu hw1-asr/csrc/
cp submission/asr/csrc/task3_gemm.cu hw1-asr/csrc/
cp submission/asr/csrc/task4_attention.cu hw1-asr/csrc/
```

Keep the other supplied framework and test files in place. Run the following commands inside the allocated GPU shell:

```bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mls-asr
mkdir -p "$HOME/nvcc-tmp"
cd hw1-asr
```

## 4. Run correctness tests

Run the full test suite:

```bash
TMPDIR="$HOME/nvcc-tmp" make test
```

The suite compares each CUDA task with a reference on small, irregular, and model-shaped inputs. Successful output ends with `ALL PASS` for each of the four tasks. The saved output is in `output-logs/asr-make-test.log`.

To run just one task while debugging, use `make test1`, `make test2`, `make test3`, or `make test4` in place of `make test`.

## 5. Run and profile the ASR model

Run the complete model with the four CUDA task groups enabled:

```bash
TMPDIR="$HOME/nvcc-tmp" ./run.sh --profile
```

The command checks the transcription and reports time per operation. The saved output is in `output-logs/asr-profile.log`. Model-loading time is reported separately and is not part of the per-transcription time.

If you need to refresh the saved logs, run these commands from `hw1-asr` in the course repository. The `submission` folder must be alongside `hw1-asr` on the shared filesystem:

```bash
set -o pipefail
TMPDIR="$HOME/nvcc-tmp" make test 2>&1 | tee ../submission/asr/output-logs/asr-make-test.log
TMPDIR="$HOME/nvcc-tmp" ./run.sh --profile 2>&1 | tee ../submission/asr/output-logs/asr-profile.log
```

Record the GPU assigned to the run with `nvidia-smi -L`; the saved GPU details are in `output-logs/gpu.txt`.

## 6. Recorded results

On the RTX 2080 Ti, all four kernel test suites passed. The full model produced the expected transcript for the supplied 3.5-second recording. Across three runs, transcription took **3345.8 +/- 7.5 ms** on average and peak GPU memory use was **8.56 GiB**.

The profile reported **82.0%** of GPU operation time in linear operations, **14.2%** in attention, and **3.4%** in RMSNorm. These are shares of profiled GPU operation time, not shares of end-to-end transcription latency. The transcript result covers the supplied test recording only; it is not an accuracy evaluation over a larger dataset.
