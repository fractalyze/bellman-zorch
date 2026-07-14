//! GPU Groth16 prover for bellman over BN256. Synthesizes the circuit with
//! bellman's constraint system, hands the witness + dense proving key to the
//! fused GPU core ([`crate::gpu::prove_prepared`] — h-FFT + the five MSMs in one
//! PJRT call), then runs bellman's exact final proof assembly on the result.
//! GPU-only: the proof is byte-identical to `groth16::create_proof`.
//!
//! The witness synthesis mirrors bellman's prover exactly (so the A/B
//! evaluation vectors and the assignment match), minus the density trackers —
//! the dense GPU MSM walks the full assignment, so per-query sparsity is
//! unneeded.

use std::ops::{AddAssign, MulAssign};

use ff::Field;
use group::{prime::PrimeCurveAffine, Curve};
use halo2curves::bn256::{Bn256, Fr, G1Affine, G2Affine, G1, G2};

use bellman::{Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};

use groth16::Proof;

use crate::gpu::{prepare, prove_prepared_timed, CoreOutputs, Phases, PreparedKey};
use crate::setup::GpuProvingKey;

/// Evaluate a linear combination against the assignment (verbatim from
/// bellman's prover, minus the density bookkeeping the dense GPU path skips).
fn eval(lc: &LinearCombination<Fr>, input_assignment: &[Fr], aux_assignment: &[Fr]) -> Fr {
    let mut acc = Fr::ZERO;
    for &(index, coeff) in lc.as_ref() {
        if coeff.is_zero_vartime() {
            continue;
        }
        let mut tmp = match index.get_unchecked() {
            Index::Input(i) => input_assignment[i],
            Index::Aux(i) => aux_assignment[i],
        };
        if coeff != Fr::ONE {
            tmp *= coeff;
        }
        acc += tmp;
    }
    acc
}

struct ProvingAssignment {
    // A and B polynomial evaluations per constraint (C is recomputed on the GPU
    // as A⊙B, so it isn't tracked here).
    a: Vec<Fr>,
    b: Vec<Fr>,
    input_assignment: Vec<Fr>,
    aux_assignment: Vec<Fr>,
}

impl ConstraintSystem<Fr> for ProvingAssignment {
    type Root = Self;

    fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.aux_assignment.push(f()?);
        Ok(Variable::new_unchecked(Index::Aux(self.aux_assignment.len() - 1)))
    }

    fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.input_assignment.push(f()?);
        Ok(Variable::new_unchecked(Index::Input(self.input_assignment.len() - 1)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, _c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Fr>) -> LinearCombination<Fr>,
        LB: FnOnce(LinearCombination<Fr>) -> LinearCombination<Fr>,
        LC: FnOnce(LinearCombination<Fr>) -> LinearCombination<Fr>,
    {
        let a = a(LinearCombination::zero());
        let b = b(LinearCombination::zero());
        self.a.push(eval(&a, &self.input_assignment, &self.aux_assignment));
        self.b.push(eval(&b, &self.input_assignment, &self.aux_assignment));
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self) {}

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

/// Generate a Groth16 proof for `circuit` with toxic-waste blinding `(r, s)`,
/// routing the FFT and all MSMs through the fused GPU core named by
/// `XLA_BELLMAN_CORE_MLIRBC`. Byte-identical to `groth16::create_proof` over
/// `gk.to_parameters()`.
pub fn create_proof<C>(
    circuit: C,
    gk: &GpuProvingKey<Bn256>,
    r: Fr,
    s: Fr,
) -> Result<Proof<Bn256>, SynthesisError>
where
    C: Circuit<Fr>,
{
    let core_path = std::env::var("XLA_BELLMAN_CORE_MLIRBC")
        .expect("set XLA_BELLMAN_CORE_MLIRBC to the bellman_core .mlirbc");
    create_proof_at(circuit, gk, r, s, &core_path)
}

/// Like [`create_proof`], but with an explicit fused-core path (compiled once,
/// then cached) — for driving several circuit shapes from one process. Uploads
/// the proving key every call; for repeated proofs of one circuit, use
/// [`prepare`] once + [`create_proof_prepared`].
pub fn create_proof_at<C>(
    circuit: C,
    gk: &GpuProvingKey<Bn256>,
    r: Fr,
    s: Fr,
    core_path: &str,
) -> Result<Proof<Bn256>, SynthesisError>
where
    C: Circuit<Fr>,
{
    create_proof_prepared(circuit, &prepare(core_path, gk), r, s)
}

/// Generate a proof reusing a [`PreparedKey`] — the dense proving key uploaded
/// to the device once — so only the per-proof witness is serialized + uploaded.
pub fn create_proof_prepared<C>(
    circuit: C,
    prepared: &PreparedKey,
    r: Fr,
    s: Fr,
) -> Result<Proof<Bn256>, SynthesisError>
where
    C: Circuit<Fr>,
{
    create_proof_prepared_timed(circuit, prepared, r, s).map(|(proof, _)| proof)
}

/// Like [`create_proof_prepared`], but also returns the host-side GPU phase
/// breakdown for the proof (profiling aid).
#[allow(clippy::many_single_char_names)]
pub fn create_proof_prepared_timed<C>(
    circuit: C,
    prepared: &PreparedKey,
    r: Fr,
    s: Fr,
) -> Result<(Proof<Bn256>, Phases), SynthesisError>
where
    C: Circuit<Fr>,
{
    let gk = prepared.gk();
    let mut prover = ProvingAssignment {
        a: vec![],
        b: vec![],
        input_assignment: vec![],
        aux_assignment: vec![],
    };

    prover.alloc_input(|| "", || Ok(Fr::ONE))?;
    circuit.synthesize(&mut prover)?;
    // Input consistency constraints (x * 0 = 0), as bellman's prover adds.
    for i in 0..prover.input_assignment.len() {
        prover.enforce(
            || "",
            |lc| lc + Variable::new_unchecked(Index::Input(i)),
            |lc| lc,
            |lc| lc,
        );
    }

    // A/B evaluation vectors, allocated at the padded domain size n and
    // populated in place (the tail stays zero) — no grow-and-realloc.
    let n = gk.h.len() + 1;
    let mut az = vec![Fr::ZERO; n];
    let mut bz = vec![Fr::ZERO; n];
    for (dst, s) in az.iter_mut().zip(&prover.a) {
        *dst = *s;
    }
    for (dst, s) in bz.iter_mut().zip(&prover.b) {
        *dst = *s;
    }
    // Full assignment [inputs ‖ aux]; reuse the inputs allocation, extend once.
    let mut z = prover.input_assignment;
    z.extend_from_slice(&prover.aux_assignment);

    let (
        CoreOutputs {
            msm_a,
            msm_b_g1,
            msm_b_g2,
            msm_l,
            msm_h,
        },
        phases,
    ) = prove_prepared_timed(prepared, &z, &az, &bz);

    // Final assembly — bellman's exact prover tail, fed the fused MSM results.
    let vk = &gk.vk;
    if bool::from(vk.delta_g1.is_identity() | vk.delta_g2.is_identity()) {
        return Err(SynthesisError::UnexpectedIdentity);
    }

    let mut g_a = vk.delta_g1 * r;
    AddAssign::<&G1Affine>::add_assign(&mut g_a, &vk.alpha_g1);
    let mut g_b = vk.delta_g2 * s;
    AddAssign::<&G2Affine>::add_assign(&mut g_b, &vk.beta_g2);
    let mut g_c;
    {
        let mut rs = r;
        rs.mul_assign(&s);
        g_c = vk.delta_g1 * rs;
        AddAssign::<&G1>::add_assign(&mut g_c, &(vk.alpha_g1 * s));
        AddAssign::<&G1>::add_assign(&mut g_c, &(vk.beta_g1 * r));
    }

    let mut a_answer = msm_a;
    AddAssign::<&G1>::add_assign(&mut g_a, &a_answer);
    MulAssign::<Fr>::mul_assign(&mut a_answer, s);
    AddAssign::<&G1>::add_assign(&mut g_c, &a_answer);

    let mut b1_answer = msm_b_g1;
    AddAssign::<&G2>::add_assign(&mut g_b, &msm_b_g2);
    MulAssign::<Fr>::mul_assign(&mut b1_answer, r);
    AddAssign::<&G1>::add_assign(&mut g_c, &b1_answer);
    AddAssign::<&G1>::add_assign(&mut g_c, &msm_h);
    AddAssign::<&G1>::add_assign(&mut g_c, &msm_l);

    let proof = Proof {
        a: g_a.to_affine(),
        b: g_b.to_affine(),
        c: g_c.to_affine(),
    };
    Ok((proof, phases))
}
