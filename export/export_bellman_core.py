"""Export the fused bellman Groth16 GPU core: h-FFT + the 5 MSMs in one
executable, using bellman's exact h pipeline (generator=7, gnark-style
coefficient tail, no bit-reverse).

The core takes the prover witness + the (dense, unfiltered) proving-key point
arrays and returns the 5 group elements bellman's final assembly needs:

    core(z_std, az_std, bz_std, A_q, Bg1_q, Bg2_q, L_q, H_q)
      -> (msm_A:G1, msm_Bg1:G1, msm_Bg2:G2, msm_L:G1, msm_h:G1)

where z_std = [inputs ‖ aux] (full assignment), az/bz = A·z / B·z evaluation
vectors (padded to n), and the queries are bellman's CRS points in dense order.

`lax.fft`/`lax.msm` lower shape-specialized, so an executable is fixed to one
(n, m, num_inputs) shape; the point VALUES are runtime inputs. Run with the
matched zkx 0.0.5 venv (see the README for the install):

    JAX_PLATFORMS=cuda,cpu .venv/bin/python \
        export/export_bellman_core.py <n> <m> <num_inputs>
"""
import io
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from zk_dtypes import bn254_g1_affine, bn254_g2_affine, bn254_sf, bn254_sf_mont, pfinfo

P = pfinfo(bn254_sf_mont).modulus  # BN254 scalar field modulus
G = 7  # halo2curves BN254 Fr MULTIPLICATIVE_GENERATOR

ART = Path(
    os.environ.get(
        "BELLMAN_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)


def _mont(int_list):
    # zk_dtypes casts each Python int (a standard-form residue in [0, P)) to its
    # Montgomery encoding directly, so the dtype does the per-element conversion.
    return jnp.array(int_list, dtype=bn254_sf_mont)


def make_core(n: int, m: int, num_inputs: int):
    shift = _mont([pow(G, i, P) for i in range(n)])
    ginv = pow(G, P - 2, P)
    inv_shift = _mont([pow(ginv, i, P) for i in range(n)])
    den = jnp.array(pow((pow(G, n, P) - 1) % P, P - 2, P), dtype=bn254_sf_mont)

    def h_fft(az_std, bz_std):
        az = lax.convert_element_type(az_std, bn254_sf_mont)
        bz = lax.convert_element_type(bz_std, bn254_sf_mont)
        cz = az * bz
        a = lax.fft(az, "IFFT", n, generator=G)
        b = lax.fft(bz, "IFFT", n, generator=G)
        c = lax.fft(cz, "IFFT", n, generator=G)
        ac = lax.fft(a * shift, "FFT", n, generator=G)
        bc = lax.fft(b * shift, "FFT", n, generator=G)
        cc = lax.fft(c * shift, "FFT", n, generator=G)
        h_evals = (ac * bc - cc) * den
        h_poly = lax.fft(h_evals, "IFFT", n, generator=G)
        return lax.convert_element_type(h_poly * inv_shift, bn254_sf)

    @jax.jit
    def core(z_std, az_std, bz_std, a_q, bg1_q, bg2_q, l_q, h_q):
        h = h_fft(az_std, bz_std)[: n - 1]
        aux = z_std[num_inputs:]
        return (
            lax.msm(z_std, a_q),
            lax.msm(z_std, bg1_q),
            lax.msm(z_std, bg2_q),
            lax.msm(aux, l_q),
            lax.msm(h, h_q),
        )

    return core


def write_bytecode(lowered, path):
    m = lowered.compiler_ir(dialect="stablehlo")
    try:
        from jax._src.interpreters import mlir as _jmlir

        data = _jmlir.module_to_bytecode(m)
    except Exception:
        buf = io.BytesIO()
        m.operation.write_bytecode(buf)
        data = buf.getvalue()
    Path(path).write_bytes(data)


def main():
    n, m, num_inputs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    num_aux = m - num_inputs
    core = make_core(n, m, num_inputs)
    args = (
        np.zeros(m, dtype=bn254_sf),  # z_std
        np.zeros(n, dtype=bn254_sf),  # az
        np.zeros(n, dtype=bn254_sf),  # bz
        np.zeros(m, dtype=bn254_g1_affine),  # A_q
        np.zeros(m, dtype=bn254_g1_affine),  # Bg1_q
        np.zeros(m, dtype=bn254_g2_affine),  # Bg2_q
        np.zeros(num_aux, dtype=bn254_g1_affine),  # L_q
        np.zeros(n - 1, dtype=bn254_g1_affine),  # H_q
    )
    lowered = core.lower(*args)
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"bellman_core_n{n}_m{m}_i{num_inputs}.mlirbc"
    write_bytecode(lowered, out)
    print(f"wrote {out} ({out.stat().st_size} B); n={n} m={m} num_inputs={num_inputs}")


if __name__ == "__main__":
    main()
