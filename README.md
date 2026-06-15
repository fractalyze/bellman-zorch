# bellman-zorch

A GPU Groth16 prover for [`bellman`](https://github.com/zkcrypto/bellman): it
synthesizes the circuit with bellman's constraint system, runs the **entire
proving back-end — the h-polynomial FFT and all five MSMs — on the zkx GPU
plugin in one fused PJRT call**, then finishes with bellman's exact proof
assembly on the CPU. Over **BN256** (= alt_bn128 = BN254, the zkx kernels'
curve), so it drives a real-world prover (the one behind Zcash Sapling /
Filecoin) with a compiler-generated GPU core instead of hand-written `ec-gpu`
kernels. The proof is **byte-identical** to `groth16::create_proof`.

## Setup

Needs an NVIDIA GPU (CUDA), a Rust toolchain, `clang`/`libclang` (the in-tree
`zkx-pjrt` shim generates its PJRT bindings with `bindgen` at build time),
Python 3.11, and [`uv`](https://docs.astral.sh/uv/). The crate has one external
path dep — `bellman` (the zkcrypto 0.14 mainline) — expected next to this repo:

```bash
git clone https://github.com/zkcrypto/bellman ../bellman   # the ../bellman path dep
```

Install the matched zkx 0.0.5 GPU plugin from the public Fractalyze package
index (this provides `jax_plugins/zkx_gpu/pjrt_c_api_gpu_plugin.so` plus the
`lax.fft`/`lax.msm` jax fork used by the exporter):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv --index-strategy unsafe-best-match \
  --index-url https://fractalyze.github.io/pypi/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  jax==0.0.5.dev20260409061337 jaxlib==0.0.5.dev20260409061337 \
  zkx-cuda-pjrt==0.0.5.dev20260409061337 zk-dtypes==0.0.4 numpy==2.4.3
```

Point the env vars at that venv — copy-paste from the repo root:

```bash
export ZKX_VENV_PYTHON=$PWD/.venv/bin/python
export ZKX_PJRT_PLUGIN=$PWD/.venv/lib/python3.11/site-packages/jax_plugins/zkx_gpu/pjrt_c_api_gpu_plugin.so
```

## Running

```bash
# CPU sanity, no GPU needed:
cargo test

# GPU byte-match on the 322-round MiMC (export its core, then run):
JAX_PLATFORMS=cuda,cpu "$ZKX_VENV_PYTHON" export/export_bellman_core.py 1024 647 2
ZKX_BELLMAN_CORE_MLIRBC=$PWD/artifacts/bellman_core_n1024_m647_i2.mlirbc \
    cargo test --test gpu_mimc -- --ignored

# Benchmark sweep (exports a core per size, then runs examples/bench.rs):
bash bench.sh                                          # default 2^13..2^18
```

The MiMC test prints its `(n, m, num_inputs)` shape, so a new circuit is just
"read the shape, export the core" — the Rust is unchanged.

## Usage

```rust
// Dense CRS. (Setup is Wnaf-free — halo2curves BN256 doesn't implement
// group::WnafGroup; `gk.to_parameters()` gives a bellman `Parameters` too.)
let gk = bellman_zorch::setup::generate_random_gpu_key::<Bn256, _, _>(circuit, rng)?;

// One proof — ZKX_BELLMAN_CORE_MLIRBC points at a core exported for this shape:
let proof = bellman_zorch::prove::create_proof(circuit, &gk, r, s)?;

// Many proofs of one circuit — upload the proving key to the device once:
let pk = bellman_zorch::gpu::prepare(core_path, &gk);
let proof = bellman_zorch::prove::create_proof_prepared(circuit, &pk, r, s)?;
```

The back-end is one exported executable, `bellman_core`
([`export/export_bellman_core.py`](export/export_bellman_core.py)) — the h-FFT
(`lax.fft`, bellman's exact convention: generator 7, no bit-reverse) and the five
`lax.msm`s, fused. `lax.fft`/`lax.msm` lower shape-specialized, so each core is
fixed to one `(n, m, num_inputs)` shape and the proving-key points are runtime
inputs.

## Benchmark

GPU core vs multi-threaded bellman CPU (RTX 5090, MiMC sweep; each size's first
GPU proof is asserted byte-identical). GPU = key uploaded once, reused:

 | rounds |      n | CPU ms | GPU/key | GPU/once | speedup|
 | ------ | ------ | ------ | ------- | -------- | ------ |
 |   4000 |   8192 | 30.65  | 106.33  | 104.65   | 0.29x. |
 |   8000 |  16384 | 54.31  | 119.62  | 116.55   | 0.47x. |
 |  16000 |  32768 | 105.80 | 114.33  | 107.53   | 0.98x. |
 |  32000 |  65536 | 191.87 | 128.21  | 114.89   | 1.67x. |
 |  64000 | 131072 | 365.69 | 153.29  | 128.50   | 2.85x. |
 | 130000 | 262144 | 688.12 | 217.18  | 161.52   | 4.26x. |

### GPU/once phase breakdown (ms/proof)

| n      | total  | serial | h2d  | dispatch | d2h  | rest  |
| ------ | ------ | ------ | ---- | -------- | ---- | ----- |
| 8192   | 104.65 | 0.23   | 0.17 | 102.48   | 0.17 | 1.60  |
| 16384  | 116.55 | 0.47   | 0.23 | 113.31   | 0.17 | 2.37  |
| 32768  | 107.53 | 0.93   | 0.37 | 102.26   | 0.21 | 3.75  |
| 65536  | 114.89 | 2.56   | 0.69 | 104.59   | 0.20 | 6.85  |
| 131072 | 128.50 | 3.70   | 1.17 | 110.60   | 0.23 | 12.80 |
| 262144 | 161.52 | 7.43   | 2.22 | 126.38   | 0.25 | 25.24 |
