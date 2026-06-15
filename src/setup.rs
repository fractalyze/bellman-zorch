//! Trusted-setup (CRS generation) for the BN256 Groth16 instance.
//!
//! This is a Wnaf-free re-implementation of bellman's
//! `groth16::generate_random_parameters`. bellman's own setup bounds the curve
//! groups by `group::WnafGroup` (a fixed-base scalar-mult speedup), which
//! halo2curves' BN256 points do not implement — and which the orphan rule
//! forbids us from adding. The QAP math here is identical to bellman's; only
//! the fixed-base Wnaf exponentiations are replaced by plain `base * scalar`,
//! and the worker-parallel loops are collapsed to single-threaded ones (setup
//! is not the part this PoC accelerates). The resulting `Parameters` are a
//! valid CRS: proofs produced with bellman's `create_proof` over them verify.

use std::ops::{AddAssign, MulAssign};
use std::sync::Arc;

use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve, Group};
use pairing::Engine;
use rand_core::RngCore;

use bellman::domain::{EvaluationDomain, Scalar};
use bellman::multicore::Worker;
use bellman::{Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};

use groth16::{Parameters, VerifyingKey};

/// Dense (unfiltered) proving key for the GPU fused core. Unlike bellman's
/// `Parameters`, the A/B queries keep their identity entries, so the core can
/// run a single dense MSM over the full assignment — an identity base
/// contributes nothing, so the result equals bellman's density-filtered MSM.
pub struct GpuProvingKey<E: Engine> {
    pub vk: VerifyingKey<E>,
    pub h: Vec<E::G1Affine>,    // length n-1
    pub l: Vec<E::G1Affine>,    // length num_aux
    pub a: Vec<E::G1Affine>,    // length num_inputs + num_aux (unfiltered)
    pub b_g1: Vec<E::G1Affine>, // length num_inputs + num_aux (unfiltered)
    pub b_g2: Vec<E::G2Affine>, // length num_inputs + num_aux (unfiltered)
    pub num_inputs: usize,
}

impl<E: Engine> GpuProvingKey<E> {
    /// bellman `Parameters` (A/B queries filtered of identity points), for
    /// driving `groth16::create_proof` as the CPU oracle.
    pub fn to_parameters(&self) -> Parameters<E> {
        Parameters {
            vk: self.vk.clone(),
            h: Arc::new(self.h.clone()),
            l: Arc::new(self.l.clone()),
            a: Arc::new(filter_identity(&self.a)),
            b_g1: Arc::new(filter_identity(&self.b_g1)),
            b_g2: Arc::new(filter_identity(&self.b_g2)),
        }
    }
}

/// Drop identity (point-at-infinity) entries — bellman's CRS omits them.
fn filter_identity<C: PrimeCurveAffine>(v: &[C]) -> Vec<C> {
    v.iter().copied().filter(|e| bool::from(!e.is_identity())).collect()
}

/// Generate a random dense CRS for `circuit` over the pairing engine `E`.
pub fn generate_random_gpu_key<E, C, R>(
    circuit: C,
    mut rng: &mut R,
) -> Result<GpuProvingKey<E>, SynthesisError>
where
    E: Engine,
    C: Circuit<E::Fr>,
    R: RngCore,
{
    let g1 = E::G1::random(&mut rng);
    let g2 = E::G2::random(&mut rng);
    let alpha = E::Fr::random(&mut rng);
    let beta = E::Fr::random(&mut rng);
    let gamma = E::Fr::random(&mut rng);
    let delta = E::Fr::random(&mut rng);
    let tau = E::Fr::random(&mut rng);

    generate_parameters::<E, C>(circuit, g1, g2, alpha, beta, gamma, delta, tau)
}

/// Generate a random CRS as bellman `Parameters` (filtered). Drop-in for
/// `groth16::generate_random_parameters`, minus the `WnafGroup` bound.
pub fn generate_random_parameters<E, C, R>(
    circuit: C,
    rng: &mut R,
) -> Result<Parameters<E>, SynthesisError>
where
    E: Engine,
    C: Circuit<E::Fr>,
    R: RngCore,
{
    Ok(generate_random_gpu_key::<E, C, R>(circuit, rng)?.to_parameters())
}

/// Circuit → QAP synthesis structure (verbatim from bellman's generator: this
/// must match bellman's `create_proof` synthesis exactly or proofs won't
/// verify).
struct KeypairAssembly<Scalar: PrimeField> {
    num_inputs: usize,
    num_aux: usize,
    num_constraints: usize,
    at_inputs: Vec<Vec<(Scalar, usize)>>,
    bt_inputs: Vec<Vec<(Scalar, usize)>>,
    ct_inputs: Vec<Vec<(Scalar, usize)>>,
    at_aux: Vec<Vec<(Scalar, usize)>>,
    bt_aux: Vec<Vec<(Scalar, usize)>>,
    ct_aux: Vec<Vec<(Scalar, usize)>>,
}

impl<Scalar: PrimeField> ConstraintSystem<Scalar> for KeypairAssembly<Scalar> {
    type Root = Self;

    fn alloc<F, A, AR>(&mut self, _: A, _: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let index = self.num_aux;
        self.num_aux += 1;
        self.at_aux.push(vec![]);
        self.bt_aux.push(vec![]);
        self.ct_aux.push(vec![]);
        Ok(Variable::new_unchecked(Index::Aux(index)))
    }

    fn alloc_input<F, A, AR>(&mut self, _: A, _: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let index = self.num_inputs;
        self.num_inputs += 1;
        self.at_inputs.push(vec![]);
        self.bt_inputs.push(vec![]);
        self.ct_inputs.push(vec![]);
        Ok(Variable::new_unchecked(Index::Input(index)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
        LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    {
        fn eval<Scalar: PrimeField>(
            l: LinearCombination<Scalar>,
            inputs: &mut [Vec<(Scalar, usize)>],
            aux: &mut [Vec<(Scalar, usize)>],
            this_constraint: usize,
        ) {
            for (index, coeff) in l.as_ref() {
                match index.get_unchecked() {
                    Index::Input(id) => inputs[id].push((*coeff, this_constraint)),
                    Index::Aux(id) => aux[id].push((*coeff, this_constraint)),
                }
            }
        }

        eval(
            a(LinearCombination::zero()),
            &mut self.at_inputs,
            &mut self.at_aux,
            self.num_constraints,
        );
        eval(
            b(LinearCombination::zero()),
            &mut self.bt_inputs,
            &mut self.bt_aux,
            self.num_constraints,
        );
        eval(
            c(LinearCombination::zero()),
            &mut self.ct_inputs,
            &mut self.ct_aux,
            self.num_constraints,
        );

        self.num_constraints += 1;
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

/// Create a dense CRS from explicit toxic waste (Wnaf-free; A/B queries kept
/// unfiltered — see [`GpuProvingKey`]).
#[allow(clippy::too_many_arguments)]
fn generate_parameters<E, C>(
    circuit: C,
    g1: E::G1,
    g2: E::G2,
    alpha: E::Fr,
    beta: E::Fr,
    gamma: E::Fr,
    delta: E::Fr,
    tau: E::Fr,
) -> Result<GpuProvingKey<E>, SynthesisError>
where
    E: Engine,
    C: Circuit<E::Fr>,
{
    let mut assembly = KeypairAssembly {
        num_inputs: 0,
        num_aux: 0,
        num_constraints: 0,
        at_inputs: vec![],
        bt_inputs: vec![],
        ct_inputs: vec![],
        at_aux: vec![],
        bt_aux: vec![],
        ct_aux: vec![],
    };

    // Allocate the "one" input variable.
    assembly.alloc_input(|| "", || Ok(E::Fr::ONE))?;

    // Synthesize the circuit.
    circuit.synthesize(&mut assembly)?;

    // Input constraints to ensure full density of the IC query: x * 0 = 0.
    for i in 0..assembly.num_inputs {
        assembly.enforce(
            || "",
            |lc| lc + Variable::new_unchecked(Index::Input(i)),
            |lc| lc,
            |lc| lc,
        );
    }

    // Bases for blind evaluation of polynomials at tau.
    let powers_of_tau = vec![Scalar::<E::Fr>(E::Fr::ZERO); assembly.num_constraints];
    let mut powers_of_tau = EvaluationDomain::from_coeffs(powers_of_tau)?;

    let gamma_inverse =
        Option::<E::Fr>::from(gamma.invert()).ok_or(SynthesisError::UnexpectedIdentity)?;
    let delta_inverse =
        Option::<E::Fr>::from(delta.invert()).ok_or(SynthesisError::UnexpectedIdentity)?;

    let worker = Worker::new();

    // Fill the raw powers of tau: powers_of_tau[i] = tau^i.
    {
        let pt = powers_of_tau.as_mut();
        let mut current_tau_power = E::Fr::ONE;
        for p in pt.iter_mut() {
            p.0 = current_tau_power;
            current_tau_power.mul_assign(&tau);
        }
    }

    // H query: h[i] = g1^{(tau^i * t(tau)) / delta}, computed from the raw tau
    // powers (before the iFFT below switches them to the Lagrange basis).
    let mut coeff = powers_of_tau.z(&tau);
    coeff.mul_assign(&delta_inverse);

    let mut h = vec![E::G1Affine::identity(); powers_of_tau.as_ref().len() - 1];
    for (h, p) in h.iter_mut().zip(powers_of_tau.as_ref().iter()) {
        let mut exp = p.0;
        exp.mul_assign(&coeff);
        *h = (g1 * exp).to_affine();
    }

    // Inverse FFT to convert powers of tau to Lagrange coefficients.
    powers_of_tau.ifft(&worker);
    let powers_of_tau = powers_of_tau.into_coeffs();

    let mut a = vec![E::G1Affine::identity(); assembly.num_inputs + assembly.num_aux];
    let mut b_g1 = vec![E::G1Affine::identity(); assembly.num_inputs + assembly.num_aux];
    let mut b_g2 = vec![E::G2Affine::identity(); assembly.num_inputs + assembly.num_aux];
    let mut ic = vec![E::G1Affine::identity(); assembly.num_inputs];
    let mut l = vec![E::G1Affine::identity(); assembly.num_aux];

    // Evaluate for inputs (ext = IC query, scaled by 1/gamma).
    eval::<E>(
        g1,
        g2,
        &powers_of_tau,
        &assembly.at_inputs,
        &assembly.bt_inputs,
        &assembly.ct_inputs,
        &mut a[0..assembly.num_inputs],
        &mut b_g1[0..assembly.num_inputs],
        &mut b_g2[0..assembly.num_inputs],
        &mut ic,
        &gamma_inverse,
        &alpha,
        &beta,
    );

    // Evaluate for auxiliary variables (ext = L query, scaled by 1/delta).
    eval::<E>(
        g1,
        g2,
        &powers_of_tau,
        &assembly.at_aux,
        &assembly.bt_aux,
        &assembly.ct_aux,
        &mut a[assembly.num_inputs..],
        &mut b_g1[assembly.num_inputs..],
        &mut b_g2[assembly.num_inputs..],
        &mut l,
        &delta_inverse,
        &alpha,
        &beta,
    );

    // The L query must be fully dense (no unconstrained variables).
    for e in l.iter() {
        if e.is_identity().into() {
            return Err(SynthesisError::UnconstrainedVariable);
        }
    }

    let g1 = g1.to_affine();
    let g2 = g2.to_affine();

    let vk = VerifyingKey::<E> {
        alpha_g1: (g1 * alpha).to_affine(),
        beta_g1: (g1 * beta).to_affine(),
        beta_g2: (g2 * beta).to_affine(),
        gamma_g2: (g2 * gamma).to_affine(),
        delta_g1: (g1 * delta).to_affine(),
        delta_g2: (g2 * delta).to_affine(),
        ic,
    };

    // A/B queries are kept UNFILTERED (identities retained) for the dense GPU
    // MSM; `to_parameters` filters them when a bellman `Parameters` is needed.
    Ok(GpuProvingKey {
        vk,
        h,
        l,
        a,
        b_g1,
        b_g2,
        num_inputs: assembly.num_inputs,
    })
}

/// Evaluate the QAP polynomials at tau and lift them onto the curve. `ext` is
/// the IC (inputs) or L (aux) query, scaled by `inv` (1/gamma or 1/delta). A/B
/// entries whose QAP coefficient is zero are left as the point at infinity and
/// filtered out by the caller, exactly as bellman does.
#[allow(clippy::too_many_arguments)]
fn eval<E: Engine>(
    g1: E::G1,
    g2: E::G2,
    powers_of_tau: &[Scalar<E::Fr>],
    at: &[Vec<(E::Fr, usize)>],
    bt: &[Vec<(E::Fr, usize)>],
    ct: &[Vec<(E::Fr, usize)>],
    a: &mut [E::G1Affine],
    b_g1: &mut [E::G1Affine],
    b_g2: &mut [E::G2Affine],
    ext: &mut [E::G1Affine],
    inv: &E::Fr,
    alpha: &E::Fr,
    beta: &E::Fr,
) {
    assert_eq!(a.len(), at.len());
    assert_eq!(a.len(), bt.len());
    assert_eq!(a.len(), ct.len());
    assert_eq!(a.len(), b_g1.len());
    assert_eq!(a.len(), b_g2.len());
    assert_eq!(a.len(), ext.len());

    fn eval_at_tau<S: PrimeField>(powers_of_tau: &[Scalar<S>], p: &[(S, usize)]) -> S {
        let mut acc = S::ZERO;
        for &(ref coeff, index) in p {
            let mut n = powers_of_tau[index].0;
            n.mul_assign(coeff);
            acc.add_assign(&n);
        }
        acc
    }

    for i in 0..a.len() {
        // Evaluate QAP polynomials at tau.
        let mut at_i = eval_at_tau(powers_of_tau, &at[i]);
        let mut bt_i = eval_at_tau(powers_of_tau, &bt[i]);
        let ct_i = eval_at_tau(powers_of_tau, &ct[i]);

        // A query (in G1).
        if !at_i.is_zero_vartime() {
            a[i] = (g1 * at_i).to_affine();
        }

        // B query (in G1 and G2).
        if !bt_i.is_zero_vartime() {
            b_g1[i] = (g1 * bt_i).to_affine();
            b_g2[i] = (g2 * bt_i).to_affine();
        }

        // ext = (beta * A + alpha * B + C) / {gamma|delta}.
        at_i *= beta;
        bt_i *= alpha;
        let mut e = at_i;
        e.add_assign(&bt_i);
        e.add_assign(&ct_i);
        e.mul_assign(inv);
        ext[i] = (g1 * e).to_affine();
    }
}
