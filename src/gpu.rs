//! BN256 fused GPU Groth16 core: the h-FFT + the five MSMs in a single xla
//! executable (exported by `export/export_bellman_core.py`). One PJRT
//! round-trip computes every group element bellman's final assembly needs —
//! the FFT runs on the GPU too, so there is no per-MSM host↔device shuttling.
//!
//! The wire layout is the zk_dtypes boundary one (standard form, 32-byte LE; G1
//! `x‖y`, G2 `x.c0‖x.c1‖y.c0‖y.c1`; identity = all-zero). The dense (unfiltered)
//! proving key is fed straight in: an identity base contributes nothing even
//! against a non-zero scalar, so the dense MSM equals bellman's density MSM.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Group};
use halo2curves::bn256::{Bn256, Fq, Fq2, Fr, G1Affine, G2Affine, G1, G2};

// xla-pjrt is curve-agnostic, so the caller names the buffer-type tags it needs.
use xla_pjrt::sys::{
    PJRT_Buffer_Type_PJRT_Buffer_Type_BN254_G1_AFFINE as BN254_G1_AFFINE,
    PJRT_Buffer_Type_PJRT_Buffer_Type_BN254_G2_AFFINE as BN254_G2_AFFINE,
    PJRT_Buffer_Type_PJRT_Buffer_Type_BN254_SF as BN254_SF,
};

use crate::setup::GpuProvingKey;

/// The persistent GPU client plus a cache of compiled cores keyed by their
/// `.mlirbc` path (callers map a circuit shape → path via the export naming
/// convention); each is compiled once and reused.
struct Gpu {
    session: xla_pjrt::Session,
    cores: RefCell<HashMap<String, &'static xla_pjrt::Executable>>,
}

thread_local! {
    /// One [`Gpu`] per thread, leaked to `'static` (a second PJRT client in one
    /// process aborts, and tearing the client down against a live CUDA context
    /// can fault — so its destructor never runs).
    static GPU: RefCell<Option<&'static Gpu>> = const { RefCell::new(None) };
}

fn gpu() -> &'static Gpu {
    GPU.with(|cell| {
        *cell.borrow_mut().get_or_insert_with(|| {
            let session = unsafe { xla_pjrt::Session::new() };
            Box::leak(Box::new(Gpu {
                session,
                cores: RefCell::new(HashMap::new()),
            }))
        })
    })
}

/// Compile (once) and return the fused core at `path`.
fn core_for(path: &str) -> &'static xla_pjrt::Executable {
    let g = gpu();
    if let Some(exe) = g.cores.borrow().get(path) {
        return exe;
    }
    let code = std::fs::read(path).unwrap_or_else(|_| panic!("read bellman_core bytecode at {path}"));
    let exe = Box::leak(Box::new(unsafe { g.session.compile(&code) }));
    g.cores.borrow_mut().insert(path.to_string(), exe);
    exe
}

/// Fail fast with a clear message if the core at `path` was exported for a
/// different circuit shape — otherwise a mismatch only surfaces as an opaque
/// PJRT abort. Best-effort: only checks convention-named artifacts.
fn assert_core_shape(path: &str, n: usize, m: usize, num_inputs: usize) {
    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if name.starts_with("bellman_core_n") {
        let expected = format!("bellman_core_n{n}_m{m}_i{num_inputs}");
        assert_eq!(
            name, expected,
            "loaded core {name:?} does not match the circuit shape {expected:?} — \
             re-export: export/export_bellman_core.py {n} {m} {num_inputs}"
        );
    }
}

const SF: usize = 32; // scalar / coordinate bytes (zk_dtypes 32-byte LE limb)
const G1B: usize = 2 * SF; // G1 affine: x ‖ y
const G2B: usize = 4 * SF; // G2 affine: x.c0 ‖ x.c1 ‖ y.c0 ‖ y.c1

/// The five group elements bellman's final proof assembly consumes.
pub struct CoreOutputs {
    pub msm_a: G1,    // A · z
    pub msm_b_g1: G1, // B · z in G1
    pub msm_b_g2: G2, // B · z in G2
    pub msm_l: G1,    // L (aux ext) query
    pub msm_h: G1,    // h · H query
}

/// The dense proving key uploaded to the device once, bound to its compiled
/// core. Reuse it across many proofs of the same circuit so the key (the bulk
/// of the per-proof payload) is serialized + transferred only once, not every
/// proof.
pub struct PreparedKey<'a> {
    gk: &'a GpuProvingKey<Bn256>,
    exe: &'static xla_pjrt::Executable,
    key: [xla_pjrt::Buffer; 5], // resident A_q, Bg1_q, Bg2_q, L_q, H_q
    m: usize,
    n: usize,
}

impl PreparedKey<'_> {
    pub fn gk(&self) -> &GpuProvingKey<Bn256> {
        self.gk
    }
}

/// Compile/fetch the core and upload the dense proving key to the device once.
pub fn prepare<'a>(core_path: &str, gk: &'a GpuProvingKey<Bn256>) -> PreparedKey<'a> {
    let m = gk.a.len();
    let num_aux = gk.l.len();
    let n = gk.h.len() + 1;
    assert_core_shape(core_path, n, m, gk.num_inputs);
    let exe = core_for(core_path);
    let s = gpu();
    let key = unsafe {
        [
            s.session.input_buffer(&g1_array(&gk.a), &[m as i64], BN254_G1_AFFINE),
            s.session.input_buffer(&g1_array(&gk.b_g1), &[m as i64], BN254_G1_AFFINE),
            s.session.input_buffer(&g2_array(&gk.b_g2), &[m as i64], BN254_G2_AFFINE),
            s.session.input_buffer(&g1_array(&gk.l), &[num_aux as i64], BN254_G1_AFFINE),
            s.session.input_buffer(&g1_array(&gk.h), &[(n - 1) as i64], BN254_G1_AFFINE),
        ]
    };
    PreparedKey { gk, exe, key, m, n }
}

/// Per-proof host-side phase timings for the fused core (profiling aid).
#[derive(Clone, Copy, Default)]
pub struct Phases {
    pub serialize: Duration, // CPU: per-proof witness (z/az/bz) -> bytes
    pub h2d: Duration,       // upload the witness
    pub dispatch: Duration,  // execute (enqueue)
    pub readback: Duration,  // read the 5 outputs (blocks on the computation)
}

/// Run the fused core with `p`'s resident key; only the per-proof inputs
/// (`z` = full assignment `[inputs ‖ aux]`, `az`/`bz` = A·z / B·z padded to n)
/// are serialized and uploaded.
pub fn prove_prepared(p: &PreparedKey, z: &[Fr], az: &[Fr], bz: &[Fr]) -> CoreOutputs {
    prove_prepared_timed(p, z, az, bz).0
}

/// Like [`prove_prepared`], but also returns the host-side phase breakdown.
pub fn prove_prepared_timed(
    p: &PreparedKey,
    z: &[Fr],
    az: &[Fr],
    bz: &[Fr],
) -> (CoreOutputs, Phases) {
    assert_eq!(z.len(), p.m, "assignment length must match the A query");
    assert_eq!(az.len(), p.n, "A·z must be padded to the domain size");
    assert_eq!(bz.len(), p.n, "B·z must be padded to the domain size");
    let s = gpu();
    let (m, n) = (p.m as i64, p.n as i64);

    let t = Instant::now();
    let (zb, azb, bzb) = (scalar_bytes(z), scalar_bytes(az), scalar_bytes(bz));
    let serialize = t.elapsed();

    let t = Instant::now();
    let z_buf = unsafe { s.session.input_buffer(&zb, &[m], BN254_SF) };
    let az_buf = unsafe { s.session.input_buffer(&azb, &[n], BN254_SF) };
    let bz_buf = unsafe { s.session.input_buffer(&bzb, &[n], BN254_SF) };
    let h2d = t.elapsed();

    // Order must match `core(z, az, bz, A_q, Bg1_q, Bg2_q, L_q, H_q)`.
    let inputs = [
        &z_buf, &az_buf, &bz_buf, &p.key[0], &p.key[1], &p.key[2], &p.key[3], &p.key[4],
    ];
    let (outs, dispatch, readback) = unsafe { s.session.run_buffers_timed(p.exe, &inputs, 5) };

    let out = CoreOutputs {
        msm_a: g1_from_bytes(&outs[0]),
        msm_b_g1: g1_from_bytes(&outs[1]),
        msm_b_g2: g2_from_bytes(&outs[2]),
        msm_l: g1_from_bytes(&outs[3]),
        msm_h: g1_from_bytes(&outs[4]),
    };
    (out, Phases { serialize, h2d, dispatch, readback })
}

// --- halo2curves BN256 ↔ zk_dtypes byte layout (standard form, 32-byte LE) ---

fn scalar_bytes(v: &[Fr]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * SF);
    for s in v {
        b.extend_from_slice(&s.to_repr());
    }
    b
}
fn g1_array(v: &[G1Affine]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * G1B);
    for p in v {
        b.extend_from_slice(&g1_to_bytes(p));
    }
    b
}
fn g2_array(v: &[G2Affine]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * G2B);
    for p in v {
        b.extend_from_slice(&g2_to_bytes(p));
    }
    b
}

fn fq_at(b: &[u8], i: usize) -> Fq {
    let mut a = [0u8; SF];
    a.copy_from_slice(&b[i * SF..(i + 1) * SF]);
    Fq::from_repr(a).expect("on-the-wire coordinate is a valid Fq")
}
fn is_identity_bytes(b: &[u8]) -> bool {
    b.iter().all(|&v| v == 0)
}

fn g1_to_bytes(p: &G1Affine) -> [u8; G1B] {
    let mut out = [0u8; G1B];
    // halo2curves' affine identity is (0, 1), not (0, 0) — map it to all-zero.
    if bool::from(p.is_identity()) {
        return out;
    }
    out[..SF].copy_from_slice(&p.x.to_repr());
    out[SF..].copy_from_slice(&p.y.to_repr());
    out
}
fn g2_to_bytes(p: &G2Affine) -> [u8; G2B] {
    let mut out = [0u8; G2B];
    if bool::from(p.is_identity()) {
        return out;
    }
    out[..SF].copy_from_slice(&p.x.c0.to_repr());
    out[SF..2 * SF].copy_from_slice(&p.x.c1.to_repr());
    out[2 * SF..3 * SF].copy_from_slice(&p.y.c0.to_repr());
    out[3 * SF..].copy_from_slice(&p.y.c1.to_repr());
    out
}
fn g1_from_bytes(b: &[u8]) -> G1 {
    if is_identity_bytes(b) {
        return G1::identity();
    }
    G1Affine {
        x: fq_at(b, 0),
        y: fq_at(b, 1),
    }
    .to_curve()
}
fn g2_from_bytes(b: &[u8]) -> G2 {
    if is_identity_bytes(b) {
        return G2::identity();
    }
    G2Affine {
        x: Fq2 {
            c0: fq_at(b, 0),
            c1: fq_at(b, 1),
        },
        y: Fq2 {
            c0: fq_at(b, 2),
            c1: fq_at(b, 3),
        },
    }
    .to_curve()
}
