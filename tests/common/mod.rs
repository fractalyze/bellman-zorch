//! Shared MiMC demo circuit, copied from bellman's own groth16 test suite
//! (`groth16/tests/common/mod.rs`). It is generic over `ff::PrimeField`, so it
//! instantiates over BN256 unchanged. Kept here as the common circuit for both
//! the Slice 0 CPU oracle and the Slice 1 GPU-prover validation test.

// Each test binary pulls in this shared module but uses only the circuits it
// needs, so some items look "dead" per-binary.
#![allow(dead_code)]

use ff::PrimeField;

use bellman::{Circuit, ConstraintSystem, SynthesisError};
use groth16::Proof;

/// Serialize a `Proof` to its canonical bytes, for byte-identity comparisons.
pub fn proof_bytes<E: pairing::Engine>(proof: &Proof<E>) -> Vec<u8> {
    let mut bytes = vec![];
    proof.write(&mut bytes).unwrap();
    bytes
}

pub const MIMC_ROUNDS: usize = 322;

/// MiMC `LongsightF322p3` (http://eprint.iacr.org/2016/492):
///
/// ```text
/// function LongsightF322p3(xL, xR) {
///     for i from 0 up to 321 {
///         xL, xR := xR + (xL + Ci)^3, xL
///     }
///     return xL
/// }
/// ```
pub fn mimc<S: PrimeField>(mut xl: S, mut xr: S, constants: &[S]) -> S {
    assert_eq!(constants.len(), MIMC_ROUNDS);

    for c in constants {
        let mut tmp1 = xl;
        tmp1.add_assign(c);
        let mut tmp2 = tmp1.square();
        tmp2.mul_assign(&tmp1);
        tmp2.add_assign(&xr);
        xr = xl;
        xl = tmp2;
    }

    xl
}

/// Demo circuit proving knowledge of a MiMC-hash preimage.
#[allow(clippy::upper_case_acronyms)]
pub struct MiMCDemo<'a, S: PrimeField> {
    pub xl: Option<S>,
    pub xr: Option<S>,
    pub constants: &'a [S],
}

impl<'a, S: PrimeField> Circuit<S> for MiMCDemo<'a, S> {
    fn synthesize<CS: ConstraintSystem<S>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        assert_eq!(self.constants.len(), MIMC_ROUNDS);

        let mut xl_value = self.xl;
        let mut xl = cs.alloc(
            || "preimage xl",
            || xl_value.ok_or(SynthesisError::AssignmentMissing),
        )?;

        let mut xr_value = self.xr;
        let mut xr = cs.alloc(
            || "preimage xr",
            || xr_value.ok_or(SynthesisError::AssignmentMissing),
        )?;

        for i in 0..MIMC_ROUNDS {
            // xL, xR := xR + (xL + Ci)^3, xL
            let cs = &mut cs.namespace(|| format!("round {}", i));

            // tmp = (xL + Ci)^2
            let tmp_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.square()
            });
            let tmp = cs.alloc(
                || "tmp",
                || tmp_value.ok_or(SynthesisError::AssignmentMissing),
            )?;

            cs.enforce(
                || "tmp = (xL + Ci)^2",
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + tmp,
            );

            // new_xL = xR + (xL + Ci)^3 = xR + tmp * (xL + Ci)
            let new_xl_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.mul_assign(&tmp_value.unwrap());
                e.add_assign(&xr_value.unwrap());
                e
            });

            let new_xl = if i == (MIMC_ROUNDS - 1) {
                // Last round: xL is the image, allocated as the public input.
                cs.alloc_input(
                    || "image",
                    || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            } else {
                cs.alloc(
                    || "new_xl",
                    || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            };

            cs.enforce(
                || "new_xL = xR + (xL + Ci)^3",
                |lc| lc + tmp,
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + new_xl - xr,
            );

            // xR = xL
            xr = xl;
            xr_value = xl_value;

            // xL = new_xL
            xl = new_xl;
            xl_value = new_xl_value;
        }

        Ok(())
    }
}
