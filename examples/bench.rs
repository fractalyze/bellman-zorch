//! Benchmark: bellman CPU `groth16::create_proof` vs the fused GPU core, across
//! MiMC circuit sizes. Each size needs its core exported first; run once to get
//! the export commands it prints, export, then run again for the GPU numbers.
//!
//! Warm-up excludes the one-time GPU client init + core compile; reported times
//! are the steady-state mean over K full proofs (synthesis + FFT + MSMs +
//! assembly) for both sides. The first GPU proof per size is asserted
//! byte-identical to the CPU proof.
//!
//!   export ZKX_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so
//!   cargo run --release --example bench

use std::ops::{AddAssign, MulAssign};
use std::path::Path;
use std::time::{Duration, Instant};

use ff::Field;
use halo2curves::bn256::{Bn256, Fr};
use rand::thread_rng;

use bellman::{Circuit, ConstraintSystem, SynthesisError};
use groth16::{create_proof as cpu_create_proof, Proof};

use bellman_zorch::gpu::prepare;
use bellman_zorch::prove::{create_proof_at, create_proof_prepared_timed};
use bellman_zorch::setup::generate_random_gpu_key;

/// MiMC `LongsightF*p3`, parametric over the number of rounds (= constants len).
struct MiMC<'a> {
    xl: Option<Fr>,
    xr: Option<Fr>,
    constants: &'a [Fr],
}

impl<'a> Circuit<Fr> for MiMC<'a> {
    fn synthesize<CS: ConstraintSystem<Fr>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let rounds = self.constants.len();
        let mut xl_value = self.xl;
        let mut xl = cs.alloc(|| "xl", || xl_value.ok_or(SynthesisError::AssignmentMissing))?;
        let mut xr_value = self.xr;
        let mut xr = cs.alloc(|| "xr", || xr_value.ok_or(SynthesisError::AssignmentMissing))?;

        for i in 0..rounds {
            let cs = &mut cs.namespace(|| format!("round {i}"));
            let tmp_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.square()
            });
            let tmp = cs.alloc(|| "tmp", || tmp_value.ok_or(SynthesisError::AssignmentMissing))?;
            cs.enforce(
                || "tmp = (xL + Ci)^2",
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + tmp,
            );
            let new_xl_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.mul_assign(&tmp_value.unwrap());
                e.add_assign(&xr_value.unwrap());
                e
            });
            let new_xl = if i == rounds - 1 {
                cs.alloc_input(|| "image", || new_xl_value.ok_or(SynthesisError::AssignmentMissing))?
            } else {
                cs.alloc(|| "new_xl", || new_xl_value.ok_or(SynthesisError::AssignmentMissing))?
            };
            cs.enforce(
                || "new_xL = xR + (xL + Ci)^3",
                |lc| lc + tmp,
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + new_xl - xr,
            );
            xr = xl;
            xr_value = xl_value;
            xl = new_xl;
            xl_value = new_xl_value;
        }
        Ok(())
    }
}

fn proof_bytes(p: &Proof<Bn256>) -> Vec<u8> {
    let mut b = vec![];
    p.write(&mut b).unwrap();
    b
}

fn mean_ms(mut f: impl FnMut(), k: u32) -> f64 {
    let t = Instant::now();
    for _ in 0..k {
        f();
    }
    t.elapsed().as_secs_f64() * 1000.0 / k as f64
}

fn main() {
    // MiMC rounds -> domain n: 2*rounds+2 constraints, n = next_pow2. Take the
    // round counts from argv, else the default 2^13..2^18 sweep.
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let rounds_list = if args.is_empty() {
        vec![4000usize, 8000, 16000, 32000, 64000, 130000]
    } else {
        args
    };
    const K: u32 = 10;
    let mut rng = thread_rng();
    // Experiment escape hatch: the jax-0.10 (lax.ntt) core isn't byte-identical
    // yet (its FFT output ordering differs), but the ntt+msm compute is the same,
    // so we still want timings. BENCH_ALLOW_MISMATCH warns instead of asserting.
    let strict = std::env::var("BENCH_ALLOW_MISMATCH").is_err();

    // GPU/key: key re-uploaded every proof. GPU/once: key uploaded once
    // (prepare) then reused. speedup = CPU / (GPU/once).
    println!(
        "{:>7} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "rounds", "n", "CPU ms", "GPU/key", "GPU/once", "speedup"
    );
    // (n, GPU/once total, serialize, h2d, dispatch, readback) ms/proof.
    let mut breakdown: Vec<(usize, f64, f64, f64, f64, f64)> = vec![];
    for &rounds in &rounds_list {
        let constants: Vec<Fr> = (0..rounds).map(|_| Fr::random(&mut rng)).collect();
        let gk = generate_random_gpu_key::<Bn256, _, _>(
            MiMC { xl: None, xr: None, constants: &constants },
            &mut rng,
        )
        .unwrap();
        let (n, m, i) = (gk.h.len() + 1, gk.a.len(), gk.num_inputs);
        let params = gk.to_parameters();
        let core = format!(
            "{}/artifacts/bellman_core_n{n}_m{m}_i{i}.mlirbc",
            env!("CARGO_MANIFEST_DIR")
        );

        let xl = Fr::random(&mut rng);
        let xr = Fr::random(&mut rng);
        let r = Fr::random(&mut rng);
        let s = Fr::random(&mut rng);
        let circ = || MiMC { xl: Some(xl), xr: Some(xr), constants: &constants };

        // Warm up + keep the CPU proof as the byte-identity oracle.
        let cpu0 = cpu_create_proof::<Bn256, _, _>(circ(), &params, r, s).unwrap();
        let cpu_ms = mean_ms(|| {
            cpu_create_proof::<Bn256, _, _>(circ(), &params, r, s).unwrap();
        }, K);

        if !Path::new(&core).exists() {
            println!(
                "{rounds:>7} {n:>8} {cpu_ms:>10.2} {:>10} {:>10} {:>8}  (export: export_bellman_core.py {n} {m} {i})",
                "SKIP", "SKIP", "-"
            );
            continue;
        }

        // key re-uploaded every proof
        let gpu0 = create_proof_at(circ(), &gk, r, s, &core).unwrap(); // warm up + compile
        if strict {
            assert_eq!(proof_bytes(&gpu0), proof_bytes(&cpu0), "GPU != CPU at rounds={rounds}");
        } else if proof_bytes(&gpu0) != proof_bytes(&cpu0) {
            eprintln!("  [rounds={rounds}] WARN: GPU != CPU (byte-identity bypassed)");
        }
        let gpu_key = mean_ms(|| {
            create_proof_at(circ(), &gk, r, s, &core).unwrap();
        }, K);

        // key uploaded once, reused across proofs; accumulate the phase breakdown
        let prepared = prepare(&core, &gk);
        let (gpu1, _) = create_proof_prepared_timed(circ(), &prepared, r, s).unwrap(); // warm up
        if strict {
            assert_eq!(proof_bytes(&gpu1), proof_bytes(&cpu0), "GPU(prepared) != CPU at rounds={rounds}");
        }
        let mut tot = Duration::ZERO;
        let (mut ser, mut h2d, mut disp, mut rb) =
            (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
        for _ in 0..K {
            let t = Instant::now();
            let (_, ph) = create_proof_prepared_timed(circ(), &prepared, r, s).unwrap();
            tot += t.elapsed();
            ser += ph.serialize;
            h2d += ph.h2d;
            disp += ph.dispatch;
            rb += ph.readback;
        }
        let avg = |d: Duration| d.as_secs_f64() * 1000.0 / K as f64;
        let gpu_once = avg(tot);
        breakdown.push((n, gpu_once, avg(ser), avg(h2d), avg(disp), avg(rb)));

        println!(
            "{rounds:>7} {n:>8} {cpu_ms:>10.2} {gpu_key:>10.2} {gpu_once:>10.2} {:>7.2}x",
            cpu_ms / gpu_once
        );
    }

    // GPU/once host-side phase breakdown. `rest` = GPU/once total minus the GPU
    // phases = CPU witness synthesis + final proof assembly. readback blocks on
    // the computation, so it ≈ dispatch+compute; the output transfer (5 points)
    // is negligible.
    println!("\n=== GPU/once phase breakdown (ms/proof) ===");
    println!(
        "{:>8} {:>9} {:>9} {:>8} {:>9} {:>9} {:>8}",
        "n", "total", "serial", "h2d", "dispatch", "d2h", "rest"
    );
    for (n, total, ser, h2d, disp, rb) in &breakdown {
        let rest = total - ser - h2d - disp - rb;
        println!("{n:>8} {total:>9.2} {ser:>9.2} {h2d:>8.2} {disp:>9.2} {rb:>9.2} {rest:>8.2}");
    }
}
