//! `bellman-zorch` — a GPU Groth16 prover for [`bellman`], drop-in for
//! `groth16::create_proof`, that routes the prover's multi-scalar
//! multiplications to the zkx GPU plugin (via the shared `zkx-pjrt` FFI) while
//! the FFT stays on CPU. BN256 (= alt_bn128 = BN254), matching the zkx kernels.
//!
//! Built on bellman's own trait stack (`pairing::Engine`, `ff`/`group`) and its
//! `Parameters`/`multiexp`/`EvaluationDomain`, with bellman's CPU prover kept as
//! a byte-identical oracle for the GPU path.
//!
//! - [`prove`] — `create_proof`: synthesize the circuit, run the fused GPU
//!   core, then bellman's exact final assembly. GPU-only, byte-identical to
//!   `groth16::create_proof`.
//! - [`gpu`] — the fused BN256 core runner (h-FFT + the five MSMs in one PJRT
//!   call).
//! - [`setup`] — a Wnaf-free CRS generator (halo2curves BN256 lacks
//!   `group::WnafGroup`, which bellman's own setup requires) producing a dense
//!   [`setup::GpuProvingKey`].

pub mod gpu;
pub mod prove;
pub mod setup;
