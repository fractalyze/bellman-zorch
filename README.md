# bellman-zorch

A GPU Groth16 prover for [`bellman`](https://github.com/zkcrypto/bellman): it
synthesizes the circuit with bellman's constraint system, runs the **entire
proving back-end — the h-polynomial FFT and all five MSMs — on the xla GPU
plugin in one fused PJRT call**, then finishes with bellman's exact proof
assembly on the CPU. Over **BN256** (= alt_bn128 = BN254, the xla kernels'
curve), so it drives a real-world prover (the one behind Zcash Sapling /
Filecoin) with a compiler-generated GPU core instead of hand-written `ec-gpu`
kernels. The proof is **byte-identical** to `groth16::create_proof`.

## Setup

Needs an NVIDIA GPU (CUDA), a Rust toolchain, `clang`/`libclang` (the in-tree
`xla-pjrt` shim generates its PJRT bindings with `bindgen` at build time),
Python 3.11, and [`uv`](https://docs.astral.sh/uv/). The crate has one external
path dep — `bellman` (the zkcrypto 0.14 mainline) — expected next to this repo:

```bash
git clone https://github.com/zkcrypto/bellman ../bellman   # the ../bellman path dep
```

Install the matched frx 0.10 GPU stack from the public Fractalyze package
index (this provides `frx_plugins/xla_cuda12/xla_cuda_plugin.so` plus the
`lax.ntt`/`lax.msm` frx distribution used by the exporter):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv --index-strategy unsafe-best-match \
  --index-url https://fractalyze.github.io/pypi/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  frx==0.10.0.dev20260716113241 frxlib==0.10.0.dev20260716113241 \
  frx-cuda12-plugin==0.10.0.dev20260716113241 frx-cuda12-pjrt==0.10.0.dev20260716113241 \
  zk-dtypes==0.0.10 numpy==2.4.3
```

Point the env vars at that venv — copy-paste from the repo root:

```bash
export XLA_VENV_PYTHON=$PWD/.venv/bin/python
export XLA_PJRT_PLUGIN=$PWD/.venv/lib/python3.11/site-packages/frx_plugins/xla_cuda12/xla_cuda_plugin.so
```

## Running

```bash
# CPU sanity, no GPU needed:
cargo test

# GPU byte-match on the 322-round MiMC (export its core, then run):
FRX_PLATFORMS=cuda,cpu "$XLA_VENV_PYTHON" export/export_bellman_core.py 1024 647 2
XLA_BELLMAN_CORE_MLIRBC=$PWD/artifacts/bellman_core_n1024_m647_i2.mlirbc \
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

// One proof — XLA_BELLMAN_CORE_MLIRBC points at a core exported for this shape:
let proof = bellman_zorch::prove::create_proof(circuit, &gk, r, s)?;

// Many proofs of one circuit — upload the proving key to the device once:
let pk = bellman_zorch::gpu::prepare(core_path, &gk);
let proof = bellman_zorch::prove::create_proof_prepared(circuit, &pk, r, s)?;
```

The back-end is one exported executable, `bellman_core`
([`export/export_bellman_core.py`](export/export_bellman_core.py)) — the h-FFT
(`lax.ntt`, bellman's exact convention: generator 7, no bit-reverse) and the five
`lax.msm`s, fused. `lax.ntt`/`lax.msm` lower shape-specialized, so each core is
fixed to one `(n, m, num_inputs)` shape and the proving-key points are runtime
inputs.

## Benchmark

GPU core vs multi-threaded bellman CPU (RTX 5090, MiMC sweep; each size's first
GPU proof is asserted byte-identical). GPU = key uploaded once, reused:

 | rounds |      n | CPU ms | GPU/key | GPU/once | speedup|
 | ------ | ------ | ------ | ------- | -------- | ------ |
 |   4000 |   8192 | 29.09  | 109.27  | 107.57   | 0.27x. |
 |   8000 |  16384 | 53.80  | 120.64  | 117.31   | 0.46x. |
 |  16000 |  32768 | 99.73  | 116.04  | 108.90   | 0.92x. |
 |  32000 |  65536 | 187.34 | 129.70  | 115.50   | 1.62x. |
 |  64000 | 131072 | 359.21 | 153.91  | 128.71   | 2.79x. |
 | 130000 | 262144 | 674.87 | 223.47  | 161.04   | 4.19x. |

### GPU/once phase breakdown (ms/proof)

| n      | total  | serial | h2d  | dispatch | d2h  | rest  |
| ------ | ------ | ------ | ---- | -------- | ---- | ----- |
| 8192   | 107.57 | 0.23   | 0.21 | 105.29   | 0.27 | 1.57  |
| 16384  | 117.31 | 0.45   | 0.29 | 114.06   | 0.26 | 2.25  |
| 32768  | 108.90 | 1.07   | 0.47 | 103.30   | 0.28 | 3.78  |
| 65536  | 115.50 | 2.50   | 0.73 | 105.49   | 0.25 | 6.53  |
| 131072 | 128.71 | 3.60   | 1.20 | 111.39   | 0.23 | 12.29 |
| 262144 | 161.04 | 7.21   | 2.21 | 126.74   | 0.29 | 24.59 |
