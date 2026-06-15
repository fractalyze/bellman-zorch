//! Real-circuit demo: the 322-round MiMC Groth16 proof with the FFT and all
//! five MSMs on the GPU via the fused core, byte-for-byte identical to bellman's
//! CPU prover and verifying. Needs the MiMC-shaped bellman core.
//!
//! `#[ignore]`d (needs the plugin + core). Build the core once, then:
//!   JAX_PLATFORMS=cuda,cpu .venv/bin/python \
//!       export/export_bellman_core.py 1024 <m> 2     # m = num_inputs + num_aux
//!   export ZKX_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so
//!   export ZKX_BELLMAN_CORE_MLIRBC=$PWD/artifacts/bellman_core_n1024_m<m>_i2.mlirbc
//!   cargo test --test gpu_mimc -- --ignored

use ff::Field;
use halo2curves::bn256::{Bn256, Fr};
use rand::thread_rng;

use groth16::{prepare_verifying_key, verify_proof};

use bellman_zorch::prove::create_proof;
use bellman_zorch::setup::generate_random_gpu_key;

mod common;
use common::{mimc, proof_bytes, MiMCDemo, MIMC_ROUNDS};

#[test]
#[ignore = "requires the zkx GPU plugin + MiMC-shaped bellman_core (set ZKX_PJRT_PLUGIN / ZKX_BELLMAN_CORE_MLIRBC)"]
fn mimc_gpu_proof_matches_bellman_byte_for_byte() {
    let mut rng = thread_rng();

    let constants = (0..MIMC_ROUNDS)
        .map(|_| Fr::random(&mut rng))
        .collect::<Vec<_>>();

    let gk = generate_random_gpu_key::<Bn256, _, _>(
        MiMCDemo::<Fr> {
            xl: None,
            xr: None,
            constants: &constants,
        },
        &mut rng,
    )
    .unwrap();
    // Print the shape so the core can be exported to match (m = gk.a.len()).
    eprintln!("MiMC core shape: n={} m={} num_inputs={}", gk.h.len() + 1, gk.a.len(), gk.num_inputs);
    let params = gk.to_parameters();
    let pvk = prepare_verifying_key(&params.vk);

    let xl = Fr::random(&mut rng);
    let xr = Fr::random(&mut rng);
    let image = mimc(xl, xr, &constants);
    let r = Fr::random(&mut rng);
    let s = Fr::random(&mut rng);

    let circuit = |xl, xr| MiMCDemo::<Fr> {
        xl: Some(xl),
        xr: Some(xr),
        constants: &constants,
    };

    let cpu =
        groth16::create_proof::<Bn256, _, _>(circuit(xl, xr), &params, r, s).unwrap();
    let gpu = create_proof(circuit(xl, xr), &gk, r, s).unwrap();

    assert_eq!(
        proof_bytes(&gpu),
        proof_bytes(&cpu),
        "MiMC fused-core GPU proof is not byte-identical to bellman's"
    );
    assert!(verify_proof(&pvk, &gpu, &[image]).is_ok(), "MiMC GPU proof failed to verify");
}
