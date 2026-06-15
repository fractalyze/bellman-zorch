//! Slice 0 — CPU oracle. Runs bellman's stock Groth16 (trusted setup, prove,
//! verify) for the MiMC circuit over BN256 (`halo2curves`), with no GPU
//! involvement. This pins the curve instantiation against bellman's full
//! prover/verifier and gives Slice 1 a CPU reference to validate the
//! GPU-multiexp prover against byte-for-byte.

use ff::Field;
use halo2curves::bn256::{Bn256, Fr};
use rand::thread_rng;

use groth16::{create_random_proof, prepare_verifying_key, verify_proof};

// bellman's own `generate_random_parameters` bounds the curve by
// `group::WnafGroup`, which halo2curves' BN256 points don't implement; use our
// Wnaf-free CRS generator instead. The prover/verifier stay bellman's.
use bellman_zorch::setup::generate_random_parameters;

mod common;
use common::{mimc, MiMCDemo, MIMC_ROUNDS};

#[test]
fn mimc_cpu_groth16_over_bn256_proves_and_verifies() {
    let mut rng = thread_rng();

    let constants = (0..MIMC_ROUNDS)
        .map(|_| Fr::random(&mut rng))
        .collect::<Vec<_>>();

    // Trusted setup over BN256 for the (witness-free) circuit shape.
    let params = {
        let c = MiMCDemo::<Fr> {
            xl: None,
            xr: None,
            constants: &constants,
        };
        generate_random_parameters::<Bn256, _, _>(c, &mut rng).unwrap()
    };
    let pvk = prepare_verifying_key(&params.vk);

    // Prove knowledge of a MiMC preimage of `image`.
    let xl = Fr::random(&mut rng);
    let xr = Fr::random(&mut rng);
    let image = mimc(xl, xr, &constants);

    let c = MiMCDemo::<Fr> {
        xl: Some(xl),
        xr: Some(xr),
        constants: &constants,
    };
    let proof = create_random_proof(c, &params, &mut rng).unwrap();

    assert!(
        verify_proof(&pvk, &proof, &[image]).is_ok(),
        "CPU Groth16 proof over BN256 failed to verify"
    );
}
